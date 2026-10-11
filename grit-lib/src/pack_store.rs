//! Store a received packfile in the object database as a pack, with its index.
//!
//! The alternative -- crate::unpack_objects -- writes every object into its own
//! loose file. That is the wrong shape for a phone: a delta is expanded before
//! it is written, so the bytes written can be several times the pack, and iOS
//! charges the volume *and* the file churn against the app's disk-write budget.
//! The same pass also has to keep the resolved objects resident for later delta
//! bases, which is how a pull of a real repository ended up killed by the
//! per-process memory limit.
//!
//! This module keeps the packfile exactly as it arrived -- compressed, written
//! once -- and builds its index from a scan of that file. Delta bases are read
//! back from the pack on demand, so the only resident state is a bounded cache
//! of recently resolved bases.
//!
//! Two things make it cheap to do after the fact rather than while streaming:
//! the pack's name *is* the hash of its own bytes, and the pack is mapped (see
//! crate::pack::read_pack_bytes_cached), so the scan reads clean file-backed
//! pages instead of owning the pack.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use sha1::Sha1;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::objects::{HashAlgo, ObjectId, ObjectKind};
use crate::odb::Odb;
use crate::pack::read_pack_bytes_cached;
use crate::unpack_objects::apply_delta;

/// Bytes pulled from the network per iteration while spooling the pack.
const READ_CHUNK: usize = 64 * 1024;
/// Default ceiling on resolved delta-base bytes kept resident.
const DEFAULT_BASE_CACHE_BYTES: usize = 32 * 1024 * 1024;
/// Longest delta chain a single object may need before we call the pack broken.
/// Git itself refuses chains deeper than 50 (--depth); this is a safety net for
/// a corrupt pack whose back-references cycle.
const MAX_DELTA_DEPTH: usize = 4096;
/// Names are made unique per process so two concurrent fetches cannot collide.
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// How a received pack is stored.
#[derive(Debug, Clone)]
pub struct StorePackOptions {
    /// Reject a pack whose raw bytes exceed this, mirroring
    /// receive.maxInputSize. None means no limit.
    pub max_input_bytes: Option<u64>,
    /// Suppress informational output.
    pub quiet: bool,
    /// Bytes of resolved delta-base content that may stay in memory.
    pub base_cache_bytes: usize,
}

impl Default for StorePackOptions {
    fn default() -> Self {
        Self {
            max_input_bytes: None,
            quiet: true,
            base_cache_bytes: DEFAULT_BASE_CACHE_BYTES,
        }
    }
}

/// The pack that was installed, and what went into it.
#[derive(Debug, Clone)]
pub struct StoredPack {
    /// objects/pack/pack-<checksum>.pack
    pub pack_path: PathBuf,
    /// objects/pack/pack-<checksum>.idx
    pub idx_path: PathBuf,
    /// Objects named by the pack header.
    pub object_count: usize,
    /// Bytes written for the packfile itself.
    pub pack_bytes: u64,
}

/// A file that removes itself unless it is moved into place.
struct TempFile {
    path: PathBuf,
    armed: bool,
}

impl TempFile {
    fn create_in(dir: &Path, prefix: &str) -> Result<Self> {
        for _ in 0..64 {
            let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("{prefix}{}_{}", std::process::id(), seq));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(_) => return Ok(Self { path, armed: true }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(Error::Io(e)),
            }
        }
        Err(Error::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not create a temporary pack file",
        )))
    }

    fn path(&self) -> &Path {
        &self.path
    }

    /// Give up ownership of the path without removing it.
    fn keep(mut self) -> PathBuf {
        self.armed = false;
        self.path.clone()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Hash of a pack body, for whichever object format the repository uses.
enum PackHasher {
    Sha1(Box<Sha1>),
    Sha256(Box<Sha256>),
}

impl PackHasher {
    fn new(algo: HashAlgo) -> Self {
        match algo {
            HashAlgo::Sha1 => Self::Sha1(Box::new(Sha1::new())),
            HashAlgo::Sha256 => Self::Sha256(Box::new(Sha256::new())),
        }
    }

    fn update(&mut self, data: &[u8]) {
        match self {
            Self::Sha1(h) => h.update(data),
            Self::Sha256(h) => h.update(data),
        }
    }

    fn finalize(self) -> Vec<u8> {
        match self {
            Self::Sha1(h) => h.finalize().to_vec(),
            Self::Sha256(h) => h.finalize().to_vec(),
        }
    }
}

/// Store the pack arriving on reader as a pack in odb.
///
/// The stream is spooled to a temporary file in objects/pack first (the same
/// directory as the final name, so installing is a rename and never a copy),
/// then indexed. Nothing partial survives a failure: the temporary files are
/// removed on every error path.
///
/// # Errors
///
/// - Error::CorruptObject -- bad signature/version, checksum mismatch,
///   truncation, an unresolvable delta chain, or a pack that exceeds
///   StorePackOptions::max_input_bytes.
/// - Error::Io -- the stream or the filesystem failed.
pub fn store_pack_stream(
    reader: &mut dyn Read,
    odb: &Odb,
    opts: &StorePackOptions,
) -> Result<StoredPack> {
    let algo = odb.hash_algo();
    let hash_len = algo.len();
    let pack_dir = odb.objects_dir().join("pack");
    fs::create_dir_all(&pack_dir).map_err(Error::Io)?;

    let temp = TempFile::create_in(&pack_dir, "tmp_pack_")?;
    let (total, digest) = spool(reader, temp.path(), hash_len, algo, opts)?;
    if total < (12 + hash_len) as u64 {
        return Err(Error::CorruptObject("pack stream truncated".to_owned()));
    }

    let mapped = read_pack_bytes_cached(temp.path())?;
    let all: &[u8] = &mapped;
    if &all[0..4] != b"PACK" {
        return Err(Error::CorruptObject(
            "not a pack stream: invalid signature".to_owned(),
        ));
    }
    let version = u32::from_be_bytes([all[4], all[5], all[6], all[7]]);
    if version != 2 && version != 3 {
        return Err(Error::CorruptObject(format!(
            "unsupported pack version {version}"
        )));
    }
    let count = u32::from_be_bytes([all[8], all[9], all[10], all[11]]) as usize;
    let data_end = all.len() - hash_len;
    if all[data_end..] != digest[..] {
        return Err(Error::CorruptObject(
            "pack trailing checksum mismatch".to_owned(),
        ));
    }

    let mut objects = scan(all, 12, count, data_end, hash_len, odb)?;
    // A thin pack is completed in place while its deltas are resolved: any base
    // the sender left out (because it knew we already had it) is appended to the
    // packfile, so what gets installed stands on its own. See PackAppender.
    let mut fix = ThinFix::new(temp.path(), data_end as u64, hash_len, algo);
    resolve_deltas(all, &mut objects, odb, opts, &mut fix)?;
    let fixed = fix.finish(count)?;
    let (final_count, final_checksum, appended, final_bytes) = match fixed {
        Some(fixed) => (
            fixed.object_count,
            fixed.checksum,
            fixed.entries,
            fixed.file_bytes,
        ),
        None => (count, digest.clone(), Vec::new(), total),
    };

    let mut entries: Vec<(ObjectId, u64, u32)> = Vec::with_capacity(final_count);
    for object in &objects {
        let oid = object
            .oid
            .ok_or_else(|| Error::CorruptObject("pack contains an unresolved delta".to_owned()))?;
        entries.push((oid, object.offset, object.crc));
    }
    entries.extend(appended);
    let idx_bytes = build_idx_v2(algo, &mut entries, &final_checksum);

    let hex = hex(&final_checksum);
    let pack_final = pack_dir.join(format!("pack-{hex}.pack"));
    let idx_final = pack_dir.join(format!("pack-{hex}.idx"));

    // Write both under temporary names, then install the pack before the index:
    // a pack with no index is ignored by readers, while an index with no pack is
    // a broken pair.
    let idx_temp = TempFile::create_in(&pack_dir, "tmp_idx_")?;
    {
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(idx_temp.path())
            .map_err(Error::Io)?;
        file.write_all(&idx_bytes).map_err(Error::Io)?;
        file.sync_all().map_err(Error::Io)?;
    }

    let pack_tmp_path = temp.path().to_path_buf();
    fs::rename(&pack_tmp_path, &pack_final).map_err(Error::Io)?;
    temp.keep();
    fs::rename(idx_temp.path(), &idx_final).map_err(Error::Io)?;
    idx_temp.keep();
    drop(mapped);

    // The pack now lives under its final name, but two caches still describe the
    // world before that rename: the mapping of the temporary file (kept alive by
    // the pack-bytes cache) and any listing of objects/pack taken while the
    // fetch was running. Dropping them is what makes the objects readable by the
    // very next lookup -- a clone checks out straight from the pack it fetched.
    crate::pack::clear_pack_cache();

    Ok(StoredPack {
        pack_path: pack_final,
        idx_path: idx_final,
        object_count: final_count,
        pack_bytes: final_bytes,
    })
}

/// Copy the stream to path, hashing every byte except the trailing checksum.
fn spool(
    reader: &mut dyn Read,
    path: &Path,
    hash_len: usize,
    algo: HashAlgo,
    opts: &StorePackOptions,
) -> Result<(u64, Vec<u8>)> {
    let file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(Error::Io)?;
    let mut out = io::BufWriter::with_capacity(READ_CHUNK, file);
    let mut hasher = PackHasher::new(algo);
    // The last hash_len bytes are the pack's own checksum and must not enter the
    // hash, so they are held back until the stream ends. Holding back a handful
    // of bytes keeps the copy bounded no matter how large the pack is.
    let mut held: Vec<u8> = Vec::with_capacity(READ_CHUNK + hash_len);
    let mut buf = vec![0u8; READ_CHUNK];
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(Error::Io)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if let Some(limit) = opts.max_input_bytes {
            if total > limit {
                return Err(Error::CorruptObject(format!(
                    "pack exceeds the maximum input size of {limit} bytes"
                )));
            }
        }
        held.extend_from_slice(&buf[..n]);
        if held.len() > hash_len {
            let flush = held.len() - hash_len;
            hasher.update(&held[..flush]);
            out.write_all(&held[..flush]).map_err(Error::Io)?;
            held.drain(..flush);
        }
    }
    // The bytes held back are the pack's own trailing checksum: they must not
    // enter the hash, but they still belong in the file.
    out.write_all(&held).map_err(Error::Io)?;
    out.flush().map_err(Error::Io)?;
    out.into_inner()
        .map_err(|e| Error::Io(e.into_error()))?
        .sync_all()
        .map_err(Error::Io)?;
    Ok((total, hasher.finalize()))
}

/// A thin pack being completed in place, matching "git index-pack --fix-thin".
///
/// A thin pack omits bases the sender knows the receiver already holds. Storing
/// it as it arrived would leave a pack that only reads while those borrowed
/// bases happen to still be around: prune or repack the pack they live in and
/// the objects built on them go with it. Git appends the borrowed bases before
/// writing the index, and so does this. The trailing checksum is dropped once,
/// each borrowed base is appended as an ordinary object, the object count in the
/// pack header is corrected, and a fresh checksum is written over the result.
struct PackAppender {
    file: File,
    algo: HashAlgo,
    hash_len: usize,
    /// Offset one past the last byte written; also where the next object goes.
    position: u64,
    /// Borrowed bases appended so far.
    count: usize,
    /// Deduplication: one delta chain can borrow the same base many times.
    seen: HashMap<ObjectId, u64>,
    entries: Vec<(ObjectId, u64, u32)>,
}

/// A pack that was completed with borrowed bases, ready to be indexed.
struct FixedPack {
    object_count: usize,
    checksum: Vec<u8>,
    entries: Vec<(ObjectId, u64, u32)>,
    file_bytes: u64,
}

impl PackAppender {
    /// Start completing a pack whose body (without its trailing checksum) is
    /// body_len bytes long.
    fn new(path: &Path, body_len: u64, hash_len: usize, algo: HashAlgo) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Error::Io)?;
        // The checksum that was verified on arrival describes the pack before
        // the appends, so it goes away now and a new one is written at the end.
        file.set_len(body_len).map_err(Error::Io)?;
        file.seek(SeekFrom::Start(body_len)).map_err(Error::Io)?;
        Ok(Self {
            file,
            algo,
            hash_len,
            position: body_len,
            count: 0,
            seen: HashMap::new(),
            entries: Vec::new(),
        })
    }

    fn append(&mut self, oid: ObjectId, kind: ObjectKind, data: &[u8]) -> Result<()> {
        if self.seen.contains_key(&oid) {
            return Ok(());
        }
        let mut entry = Vec::with_capacity(data.len() / 3 + 16);
        write_object_header(&mut entry, kind, data.len());
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| Error::CorruptObject(format!("pack zlib stream: {e}")))?;
        let compressed = encoder
            .finish()
            .map_err(|e| Error::CorruptObject(format!("pack zlib stream: {e}")))?;
        entry.extend_from_slice(&compressed);
        let offset = self.position;
        self.file.write_all(&entry).map_err(Error::Io)?;
        self.position += entry.len() as u64;
        self.count += 1;
        self.entries.push((oid, offset, crc32(&entry)));
        self.seen.insert(oid, offset);
        Ok(())
    }

    /// Correct the header count, checksum the finished pack, write its trailer.
    fn finish(mut self, original_count: usize) -> Result<FixedPack> {
        let total = original_count + self.count;
        if total > u32::MAX as usize {
            return Err(Error::CorruptObject(
                "pack would hold more objects than its header can count".to_owned(),
            ));
        }
        self.file.seek(SeekFrom::Start(8)).map_err(Error::Io)?;
        self.file
            .write_all(&(total as u32).to_be_bytes())
            .map_err(Error::Io)?;
        self.file.sync_all().map_err(Error::Io)?;

        // Hash everything before the trailer, in bounded chunks: the pack is read
        // back rather than held, which is the whole point of writing it out.
        let mut hasher = PackHasher::new(self.algo);
        self.file.seek(SeekFrom::Start(0)).map_err(Error::Io)?;
        let mut buf = vec![0u8; READ_CHUNK];
        let mut left = self.position;
        while left > 0 {
            let want = buf.len().min(left as usize);
            let n = self.file.read(&mut buf[..want]).map_err(Error::Io)?;
            if n == 0 {
                return Err(Error::CorruptObject(
                    "pack ended while it was being checksummed".to_owned(),
                ));
            }
            hasher.update(&buf[..n]);
            left -= n as u64;
        }
        let checksum = hasher.finalize();
        self.file
            .seek(SeekFrom::Start(self.position))
            .map_err(Error::Io)?;
        self.file.write_all(&checksum).map_err(Error::Io)?;
        self.file.sync_all().map_err(Error::Io)?;

        Ok(FixedPack {
            object_count: total,
            checksum,
            entries: self.entries,
            file_bytes: self.position + self.hash_len as u64,
        })
    }
}

/// The in-place completion of a thin pack, created only once a borrowed base is
/// actually needed -- a pack that arrived complete is left untouched.
struct ThinFix<'a> {
    pack_path: &'a Path,
    body_len: u64,
    hash_len: usize,
    algo: HashAlgo,
    appender: Option<PackAppender>,
}

impl<'a> ThinFix<'a> {
    fn new(pack_path: &'a Path, body_len: u64, hash_len: usize, algo: HashAlgo) -> Self {
        Self {
            pack_path,
            body_len,
            hash_len,
            algo,
            appender: None,
        }
    }

    fn append(&mut self, oid: ObjectId, kind: ObjectKind, data: &[u8]) -> Result<()> {
        if self.appender.is_none() {
            let appender = PackAppender::new(self.pack_path, self.body_len, self.hash_len, self.algo)?;
            self.appender = Some(appender);
        }
        match &mut self.appender {
            Some(appender) => appender.append(oid, kind, data),
            None => Ok(()),
        }
    }

    fn finish(self, original_count: usize) -> Result<Option<FixedPack>> {
        match self.appender {
            Some(appender) => Ok(Some(appender.finish(original_count)?)),
            None => Ok(None),
        }
    }
}

/// Type/size header of one packed object, as the pack format writes it.
fn write_object_header(out: &mut Vec<u8>, kind: ObjectKind, size: usize) {
    let type_code: u8 = match kind {
        ObjectKind::Commit => 1,
        ObjectKind::Tree => 2,
        ObjectKind::Blob => 3,
        ObjectKind::Tag => 4,
    };
    let mut size = size;
    let mut byte = ((type_code & 0x7) << 4) | (size & 0x0f) as u8;
    size >>= 4;
    while size > 0 {
        out.push(byte | 0x80);
        byte = (size & 0x7f) as u8;
        size >>= 7;
    }
    out.push(byte);
}

/// One packed object, as found by the scan.
struct ScannedObject {
    offset: u64,
    /// Offset of the zlib payload (after the type/size header, and after the
    /// base reference for a delta).
    payload: usize,
    /// CRC32 of the object's raw bytes, as the index wants them.
    crc: u32,
    /// Present for delta objects: what this object's content is built from.
    base: Option<DeltaBase>,
    kind: Option<ObjectKind>,
    oid: Option<ObjectId>,
}

enum DeltaBase {
    /// OFS_DELTA: absolute offset of the base in this pack.
    Offset(u64),
    /// REF_DELTA: object id of the base.
    Oid(ObjectId),
}

/// Walk the pack's objects in file order, recording where each one starts and
/// what it is, and hashing every object that does not depend on another.
fn scan(
    all: &[u8],
    mut pos: usize,
    count: usize,
    data_end: usize,
    hash_len: usize,
    odb: &Odb,
) -> Result<Vec<ScannedObject>> {
    let mut objects = Vec::with_capacity(count);
    for _ in 0..count {
        let offset = pos as u64;
        let (type_code, _size) = parse_object_header(all, &mut pos)?;
        let mut base = None;
        let mut kind = None;
        match type_code {
            1..=4 => kind = Some(kind_of(type_code)?),
            6 => {
                let base_offset = parse_ofs_delta_base(all, &mut pos, offset)?;
                base = Some(DeltaBase::Offset(base_offset));
            }
            7 => {
                let end = pos + hash_len;
                let bytes = all.get(pos..end).ok_or_else(|| {
                    Error::CorruptObject("truncated ref-delta base oid".to_owned())
                })?;
                pos = end;
                base = Some(DeltaBase::Oid(ObjectId::from_bytes(bytes)?));
            }
            other => {
                return Err(Error::CorruptObject(format!(
                    "unknown packed-object type {other}"
                )))
            }
        }
        let payload = pos;
        let (data, consumed) = inflate_at(all, pos)?;
        pos += consumed;
        if pos > data_end {
            return Err(Error::CorruptObject(
                "pack object runs past the trailing checksum".to_owned(),
            ));
        }
        let crc = crc32(&all[offset as usize..pos]);
        let oid = kind.map(|kind| odb.hash(kind, &data));
        objects.push(ScannedObject {
            offset,
            payload,
            crc,
            base,
            kind,
            oid,
        });
    }
    if pos != data_end {
        return Err(Error::CorruptObject(
            "pack has data after its objects".to_owned(),
        ));
    }
    Ok(objects)
}

/// Resolve every delta, reading bases out of the pack instead of holding them.
///
/// A base that is not in the pack at all is a thin pack: it is read from the odb
/// and handed to fix so the pack can be completed before it is indexed.
fn resolve_deltas(
    all: &[u8],
    objects: &mut [ScannedObject],
    odb: &Odb,
    opts: &StorePackOptions,
    fix: &mut ThinFix<'_>,
) -> Result<()> {
    let by_offset: HashMap<u64, usize> = objects
        .iter()
        .enumerate()
        .map(|(index, object)| (object.offset, index))
        .collect();
    let mut by_oid: HashMap<ObjectId, usize> = objects
        .iter()
        .enumerate()
        .filter_map(|(index, object)| object.oid.map(|oid| (oid, index)))
        .collect();
    let mut cache = BaseCache::new(opts.base_cache_bytes);

    loop {
        let mut progressed = false;
        for index in 0..objects.len() {
            if objects[index].oid.is_some() {
                continue;
            }
            match &objects[index].base {
                None => continue,
                // An ofs-delta can only be resolved against this pack.
                Some(DeltaBase::Offset(offset)) if !by_offset.contains_key(offset) => continue,
                // A ref-delta may name an object the server left out because we
                // already have it (a thin pack), so it is always worth a try.
                Some(_) => {}
            }
            let Some((kind, data)) = content_of(
                all,
                objects,
                &by_offset,
                &by_oid,
                &mut cache,
                index,
                0,
                odb,
                fix,
            )?
            else {
                continue;
            };
            let oid = odb.hash(kind, &data);
            objects[index].kind = Some(kind);
            objects[index].oid = Some(oid);
            by_oid.insert(oid, index);
            progressed = true;
        }
        if objects.iter().all(|object| object.oid.is_some()) {
            return Ok(());
        }
        if !progressed {
            let unresolved = objects.iter().filter(|object| object.oid.is_none()).count();
            return Err(Error::CorruptObject(format!(
                "{unresolved} delta(s) could not be resolved"
            )));
        }
    }
}

/// Content of one scanned object, resolved through its delta chain.
///
/// None means the chain reaches a REF_DELTA whose base has not been resolved
/// yet; the caller retries once other objects settle. A REF_DELTA base that is
/// not in this pack at all is read from the odb, which is what makes a thin
/// pack (one the server trimmed because we already hold the base) work. Every
/// base that gets resolved is cached, but only within the caller's byte budget,
/// so a deep pack costs time rather than memory.
fn content_of(
    all: &[u8],
    objects: &[ScannedObject],
    by_offset: &HashMap<u64, usize>,
    by_oid: &HashMap<ObjectId, usize>,
    cache: &mut BaseCache,
    index: usize,
    depth: usize,
    odb: &Odb,
    fix: &mut ThinFix<'_>,
) -> Result<Option<(ObjectKind, Arc<Vec<u8>>)>> {
    if depth > MAX_DELTA_DEPTH {
        return Err(Error::CorruptObject(
            "delta chain is deeper than any valid pack".to_owned(),
        ));
    }
    if let Some(hit) = cache.get(objects[index].offset) {
        return Ok(Some(hit));
    }
    let (kind, data) = match &objects[index].base {
        None => {
            let kind = objects[index]
                .kind
                .ok_or_else(|| Error::CorruptObject("packed object without a type".to_owned()))?;
            let (data, _) = inflate_at(all, objects[index].payload)?;
            (kind, data)
        }
        Some(base) => {
            let resolved = match base {
                DeltaBase::Offset(offset) => match by_offset.get(offset).copied() {
                    Some(base_index) => content_of(
                        all,
                        objects,
                        by_offset,
                        by_oid,
                        cache,
                        base_index,
                        depth + 1,
                        odb,
                        fix,
                    )?,
                    None => return Ok(None),
                },
                DeltaBase::Oid(oid) => match by_oid.get(oid).copied() {
                    Some(base_index) => content_of(
                        all,
                        objects,
                        by_offset,
                        by_oid,
                        cache,
                        base_index,
                        depth + 1,
                        odb,
                        fix,
                    )?,
                    // Thin pack: the base never came over the wire because the
                    // server knows we have it. Read it from the odb *and append
                    // it to the pack*, so the pack we install does not depend on
                    // the copy we borrowed from (git index-pack --fix-thin).
                    None => match odb.read(oid) {
                        Ok(object) => {
                            fix.append(*oid, object.kind, &object.data)?;
                            Some((object.kind, Arc::new(object.data)))
                        }
                        Err(_) => None,
                    },
                },
            };
            let Some((base_kind, base_data)) = resolved else {
                return Ok(None);
            };
            let (delta, _) = inflate_at(all, objects[index].payload)?;
            (base_kind, apply_delta(base_data.as_ref(), &delta)?)
        }
    };
    let data = Arc::new(data);
    cache.put(objects[index].offset, kind, Arc::clone(&data));
    Ok(Some((kind, data)))
}

/// Bounded FIFO cache of resolved delta bases.
struct BaseCache {
    budget: usize,
    used: usize,
    entries: HashMap<u64, (ObjectKind, Arc<Vec<u8>>)>,
    order: VecDeque<u64>,
}

impl BaseCache {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            used: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&self, offset: u64) -> Option<(ObjectKind, Arc<Vec<u8>>)> {
        self.entries.get(&offset).cloned()
    }

    fn put(&mut self, offset: u64, kind: ObjectKind, data: Arc<Vec<u8>>) {
        if self.budget == 0 || data.len() > self.budget {
            return;
        }
        if let Some((_, old)) = self.entries.insert(offset, (kind, Arc::clone(&data))) {
            self.used -= old.len();
        } else {
            self.order.push_back(offset);
        }
        self.used += data.len();
        while self.used > self.budget {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some((_, evicted)) = self.entries.remove(&oldest) {
                self.used -= evicted.len();
            }
        }
    }
}

/// Build the version-2 pack index for entries.
///
/// The layout is git's: magic and version, the 256-entry fanout, the sorted
/// object names, one CRC32 per name, 4-byte offsets (with an 8-byte table
/// behind the high bit), then the pack checksum and the index's own checksum.
fn build_idx_v2(algo: HashAlgo, entries: &mut [(ObjectId, u64, u32)], pack_checksum: &[u8]) -> Vec<u8> {
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut out = Vec::with_capacity(8 + 256 * 4 + entries.len() * 28 + 64);
    out.extend_from_slice(&[0xff, b't', b'O', b'c']);
    out.extend_from_slice(&2u32.to_be_bytes());

    let mut buckets = [0u32; 256];
    for (oid, _, _) in entries.iter() {
        buckets[oid.as_bytes()[0] as usize] += 1;
    }
    let mut running = 0u32;
    for bucket in buckets {
        running += bucket;
        out.extend_from_slice(&running.to_be_bytes());
    }
    for (oid, _, _) in entries.iter() {
        out.extend_from_slice(oid.as_bytes());
    }
    for (_, _, crc) in entries.iter() {
        out.extend_from_slice(&crc.to_be_bytes());
    }
    let mut wide: Vec<u64> = Vec::new();
    for (_, offset, _) in entries.iter() {
        if *offset < 0x8000_0000 {
            out.extend_from_slice(&(*offset as u32).to_be_bytes());
        } else {
            out.extend_from_slice(&(0x8000_0000u32 | wide.len() as u32).to_be_bytes());
            wide.push(*offset);
        }
    }
    for offset in wide {
        out.extend_from_slice(&offset.to_be_bytes());
    }
    out.extend_from_slice(pack_checksum);
    let mut hasher = PackHasher::new(algo);
    hasher.update(&out);
    let digest = hasher.finalize();
    out.extend_from_slice(&digest);
    out
}

/// Type/size header of one packed object.
fn parse_object_header(bytes: &[u8], pos: &mut usize) -> Result<(u8, usize)> {
    let mut c = *bytes
        .get(*pos)
        .ok_or_else(|| Error::CorruptObject("truncated pack object header".to_owned()))?;
    *pos += 1;
    let type_code = (c >> 4) & 0x7;
    let mut size = (c & 0x0f) as usize;
    let mut shift = 4u32;
    while c & 0x80 != 0 {
        c = *bytes
            .get(*pos)
            .ok_or_else(|| Error::CorruptObject("truncated pack object header".to_owned()))?;
        *pos += 1;
        size |= ((c & 0x7f) as usize) << shift;
        shift += 7;
        if shift > 63 {
            return Err(Error::CorruptObject(
                "pack object size is out of range".to_owned(),
            ));
        }
    }
    Ok((type_code, size))
}

/// OFS_DELTA back-reference: a negative distance from this object's offset.
fn parse_ofs_delta_base(bytes: &[u8], pos: &mut usize, object_offset: u64) -> Result<u64> {
    let mut c = *bytes
        .get(*pos)
        .ok_or_else(|| Error::CorruptObject("truncated ofs-delta header".to_owned()))?;
    *pos += 1;
    let mut value = u64::from(c & 0x7f);
    while c & 0x80 != 0 {
        c = *bytes
            .get(*pos)
            .ok_or_else(|| Error::CorruptObject("truncated ofs-delta header".to_owned()))?;
        *pos += 1;
        value = ((value + 1) << 7) | u64::from(c & 0x7f);
    }
    object_offset
        .checked_sub(value)
        .ok_or_else(|| Error::CorruptObject("ofs-delta base offset underflow".to_owned()))
}

/// Inflate the zlib stream at start, reporting how many bytes it consumed.
fn inflate_at(bytes: &[u8], start: usize) -> Result<(Vec<u8>, usize)> {
    let mut decoder = flate2::bufread::ZlibDecoder::new(&bytes[start..]);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::CorruptObject(format!("pack zlib stream: {e}")))?;
    let consumed = decoder.total_in() as usize;
    Ok((out, consumed))
}

fn kind_of(type_code: u8) -> Result<ObjectKind> {
    match type_code {
        1 => Ok(ObjectKind::Commit),
        2 => Ok(ObjectKind::Tree),
        3 => Ok(ObjectKind::Blob),
        4 => Ok(ObjectKind::Tag),
        other => Err(Error::CorruptObject(format!(
            "type code {other} is not a regular object type"
        ))),
    }
}

/// CRC32 (IEEE) of an object's raw bytes, as a pack index stores it.
fn crc32(data: &[u8]) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (index, slot) in table.iter_mut().enumerate() {
            let mut value = index as u32;
            for _ in 0..8 {
                value = if value & 1 != 0 {
                    0xEDB8_8320 ^ (value >> 1)
                } else {
                    value >> 1
                };
            }
            *slot = value;
        }
        table
    });
    let mut crc = 0xFFFF_FFFFu32;
    for byte in data {
        crc = table[((crc ^ u32::from(*byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-encode a pack object header for the test packs below.
    fn header(type_code: u8, size: usize) -> Vec<u8> {
        let mut out = Vec::new();
        let mut size = size;
        let first = ((type_code & 0x7) << 4) | (size & 0x0f) as u8;
        size >>= 4;
        if size > 0 {
            out.push(first | 0x80);
            while size > 0 {
                let byte = (size & 0x7f) as u8;
                size >>= 7;
                out.push(if size > 0 { byte | 0x80 } else { byte });
            }
        } else {
            out.push(first);
        }
        out
    }

    fn zlib(data: &[u8]) -> Vec<u8> {
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).expect("compress");
        encoder.finish().expect("finish")
    }

    fn entry(type_code: u8, data: &[u8]) -> Vec<u8> {
        let mut out = header(type_code, data.len());
        out.extend_from_slice(&zlib(data));
        out
    }

    /// The inverse of parse_ofs_delta_base, as git writes it.
    fn encode_ofs(distance: usize) -> Vec<u8> {
        let mut out = vec![(distance & 0x7f) as u8];
        let mut value = distance >> 7;
        while value > 0 {
            value -= 1;
            out.push(0x80 | (value & 0x7f) as u8);
            value >>= 7;
        }
        out.reverse();
        out
    }

    fn finish(entries: Vec<Vec<u8>>) -> Vec<u8> {
        let mut pack = Vec::new();
        pack.extend_from_slice(b"PACK");
        pack.extend_from_slice(&2u32.to_be_bytes());
        pack.extend_from_slice(&(entries.len() as u32).to_be_bytes());
        for entry in entries {
            pack.extend_from_slice(&entry);
        }
        let mut hasher = Sha1::new();
        hasher.update(&pack);
        pack.extend_from_slice(&hasher.finalize());
        pack
    }

    fn make_pack(objects: &[(u8, &[u8])]) -> Vec<u8> {
        finish(
            objects
                .iter()
                .map(|(type_code, data)| entry(*type_code, data))
                .collect(),
        )
    }

    fn make_ofs_delta_pack(base: &[u8], target: &[u8]) -> (Vec<u8>, ObjectId, ObjectId) {
        let base_entry = entry(3, base);
        let delta = crate::delta_encode::encode_lcp_delta(base, target).expect("delta");
        let mut delta_entry = header(6, delta.len());
        delta_entry.extend_from_slice(&encode_ofs(base_entry.len()));
        delta_entry.extend_from_slice(&zlib(&delta));
        let pack = finish(vec![base_entry, delta_entry]);
        (
            pack,
            Odb::hash_object_data(ObjectKind::Blob, base),
            Odb::hash_object_data(ObjectKind::Blob, target),
        )
    }

    fn odb_in(dir: &Path) -> Odb {
        let objects = dir.join("objects");
        fs::create_dir_all(&objects).expect("objects dir");
        Odb::new(&objects)
    }

    /// Every entry in pack_dir, ignoring the AppleDouble sidecars ExFAT writes.
    fn dir_entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .expect("read pack dir")
            .map(|entry| entry.expect("dir entry").file_name().to_string_lossy().to_string())
            .filter(|name| !name.starts_with("._"))
            .collect();
        names.sort();
        names
    }

    #[test]
    fn ofs_delta_encoding_round_trips() {
        for distance in [1usize, 127, 128, 255, 16_384, 16_385, 1 << 20] {
            let bytes = encode_ofs(distance);
            let mut pos = 0;
            let base = parse_ofs_delta_base(&bytes, &mut pos, 1_000_000 + distance as u64)
                .expect("decode");
            assert_eq!(base, 1_000_000, "distance {distance}");
            assert_eq!(pos, bytes.len(), "distance {distance}");
        }
    }

    #[test]
    fn stores_a_pack_and_reads_its_objects_back() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let odb = odb_in(tmp.path());
        let pack = make_pack(&[(3, b"hello\n"), (3, b"world\n")]);

        let stored = store_pack_stream(
            &mut std::io::Cursor::new(pack.clone()),
            &odb,
            &StorePackOptions::default(),
        )
        .expect("store pack");

        assert_eq!(stored.object_count, 2);
        assert_eq!(stored.pack_bytes, pack.len() as u64);
        assert_eq!(
            fs::metadata(&stored.pack_path).expect("pack stat").len(),
            pack.len() as u64
        );
        assert!(stored.idx_path.exists());

        // The pack is named after the hash of its own bytes.
        let mut hasher = Sha1::new();
        hasher.update(&pack[..pack.len() - 20]);
        let digest = hasher.finalize();
        assert!(stored
            .pack_path
            .to_string_lossy()
            .ends_with(&format!("pack-{}.pack", hex(&digest))));

        for data in [b"hello\n".as_slice(), b"world\n".as_slice()] {
            let oid = Odb::hash_object_data(ObjectKind::Blob, data);
            let object = odb.read(&oid).expect("read back");
            assert_eq!(object.data, data);
        }

        // Exactly the pair, no temporaries.
        assert_eq!(
            dir_entries(&tmp.path().join("objects/pack")),
            vec![
                stored
                    .idx_path
                    .file_name()
                    .expect("idx name")
                    .to_string_lossy()
                    .to_string(),
                stored
                    .pack_path
                    .file_name()
                    .expect("pack name")
                    .to_string_lossy()
                    .to_string(),
            ]
        );

        // The index we wrote is readable by the pack reader.
        let index = crate::pack::read_pack_index(&stored.idx_path).expect("parse idx");
        assert_eq!(index.entries.len(), 2);
    }

    #[test]
    fn resolves_an_ofs_delta_from_the_stored_pack() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let odb = odb_in(tmp.path());
        let (pack, base_oid, target_oid) = make_ofs_delta_pack(b"hello world\n", b"hello brave world\n");

        let stored = store_pack_stream(
            &mut std::io::Cursor::new(pack),
            &odb,
            &StorePackOptions::default(),
        )
        .expect("store pack");

        assert_eq!(stored.object_count, 2);
        assert_eq!(odb.read(&base_oid).expect("base").data, b"hello world\n");
        assert_eq!(
            odb.read(&target_oid).expect("delta target").data,
            b"hello brave world\n"
        );
    }

    #[test]
    fn truncated_pack_leaves_nothing_behind() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let odb = odb_in(tmp.path());
        let pack = make_pack(&[(3, b"hello\n")]);
        let truncated = pack[..pack.len() - 1].to_vec();

        let err = store_pack_stream(
            &mut std::io::Cursor::new(truncated),
            &odb,
            &StorePackOptions::default(),
        );
        assert!(err.is_err(), "a truncated pack must not be stored");
        assert!(
            dir_entries(&tmp.path().join("objects/pack")).is_empty(),
            "leftovers: {:?}",
            dir_entries(&tmp.path().join("objects/pack"))
        );
    }

    #[test]
    fn oversized_pack_is_rejected_without_leaving_a_pack() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let odb = odb_in(tmp.path());
        let pack = make_pack(&[(3, b"hello\n")]);
        let opts = StorePackOptions {
            max_input_bytes: Some(4),
            ..StorePackOptions::default()
        };
        let err = store_pack_stream(&mut std::io::Cursor::new(pack), &odb, &opts);
        assert!(err.is_err());
        assert!(dir_entries(&tmp.path().join("objects/pack")).is_empty());
    }

    #[test]
    fn completes_a_thin_pack_so_it_stands_on_its_own() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let odb = odb_in(tmp.path());
        let base = b"hello world\n".as_slice();
        let target = b"hello brave world\n".as_slice();
        // The base never appears in the pack: a thin pack names it and expects
        // the receiver to hold it already.
        let base_oid = odb
            .write_local(ObjectKind::Blob, base)
            .expect("write base");
        let delta = crate::delta_encode::encode_lcp_delta(base, target).expect("delta");
        let mut delta_entry = header(7, delta.len());
        delta_entry.extend_from_slice(base_oid.as_bytes());
        delta_entry.extend_from_slice(&zlib(&delta));
        let pack = finish(vec![delta_entry]);

        let stored = store_pack_stream(
            &mut std::io::Cursor::new(pack),
            &odb,
            &StorePackOptions::default(),
        )
        .expect("store thin pack");

        // The borrowed base was appended, so the pack counts two objects and its
        // index names both of them.
        assert_eq!(stored.object_count, 2);
        let index = crate::pack::read_pack_index(&stored.idx_path).expect("parse idx");
        assert_eq!(index.entries.len(), 2);
        assert!(
            index
                .entries
                .iter()
                .any(|entry| entry.oid.as_slice() == base_oid.as_bytes()),
            "the borrowed base must be in the stored pack"
        );

        // And it still reads once the copy it borrowed from is gone: that is the
        // difference between --fix-thin and storing the pack as it arrived.
        let loose = tmp
            .path()
            .join("objects")
            .join(base_oid.loose_prefix())
            .join(base_oid.loose_suffix());
        fs::remove_file(&loose).expect("drop the borrowed copy");
        crate::pack::clear_pack_cache();
        let target_oid = Odb::hash_object_data(ObjectKind::Blob, target);
        let object = crate::pack::read_object_from_packs(&tmp.path().join("objects"), &target_oid)
            .expect("pack must be self-contained");
        assert_eq!(object.data, target);
    }
}
