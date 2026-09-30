//! Smart-HTTP Git transport over a pluggable HTTP client.
//!
//! This module ports the smart-HTTP fetch protocol from the CLI's
//! `http_smart.rs` into an embedder-shaped surface:
//!
//! * [`HttpClient`] — the minimal request surface the protocol needs: a `GET`
//!   (used for `info/refs?service=git-upload-pack` discovery) and a `POST`
//!   (used for the stateless-RPC `git-upload-pack` / `git-receive-pack`
//!   request body). Embedders supply their own client so grit-lib never forces
//!   a particular TLS / async / proxy stack on them.
//! * [`SmartHttpTransport`] — a [`Transport`] that performs the `info/refs`
//!   discovery on [`Transport::connect`] and exposes the parsed advertisement
//!   through a [`Connection`].
//! * [`http_fetch`] — drives the stateless-RPC negotiation (`want`/`have`/`done`
//!   over repeated POSTs), demultiplexes the side-band pack, ingests it with
//!   [`crate::unpack_objects`], and returns a [`crate::transfer::FetchOutcome`]
//!   — reusing the same refspec/tag/prune/classification helpers as the
//!   in-process and `git://` fetch paths.
//!
//! A default [`ureq`]-backed [`HttpClient`] lives in [`crate::transport::http::ureq_client`]
//! behind the `http-ureq` cargo feature; it wires a [`CredentialProvider`] for
//! HTTP basic auth on `401`.
//!
//! Both protocol v0/v1 (the classic stateless RPC) and protocol v2 (the
//! stateless multi-POST flow) are implemented here. A v2 server is detected from
//! the `version 2` capability advertisement returned by `info/refs` (requested
//! with the `Git-Protocol: version=2` header); [`http_fetch`] then runs the v2
//! `command=ls-refs` + `command=fetch` rounds as separate POSTs — each round
//! resends the capability echo, all `want`s, and the accumulated `have`s —
//! reusing the shared v2 request framing and side-band demuxer from
//! [`crate::fetch`].

use std::collections::{HashSet, VecDeque};
use std::io::{Cursor, Read, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::fetch::Progress;
use crate::fetch_negotiator::SkippingNegotiator;
use crate::objects::ObjectId;
use crate::pkt_line;
use crate::protocol_v2;
use crate::refspec::{parse_fetch_refspec, RefspecItem};
use crate::transfer::{
    classify_update, match_positive, open_odb, prune_tracking_refs, ref_excluded, refspecs_force,
    FetchOptions, FetchOutcome, RefUpdate, TagMode, UpdateMode,
};
use crate::transport::{Advertisement, ConnectOptions, Connection, Service, Transport};

#[cfg(feature = "http-ureq")]
pub mod ureq_client;

/// The minimal HTTP surface the smart-HTTP transport needs.
///
/// Implementations legitimately perform real network I/O; the trait makes no
/// assumption about the underlying stack (blocking/async, TLS provider, proxy,
/// cookies), so an embedder can route Git's HTTP through whatever client it
/// already uses.
///
/// The `git_protocol` argument carries the value of the `Git-Protocol` request
/// header (e.g. `version=2`) when the caller wants to negotiate a protocol
/// version; pass it through verbatim. A default `Git-Protocol` for every request
/// may be supplied via [`HttpClient::git_protocol_header`].
pub trait HttpClient: Send + Sync {
    /// Issue a `GET` to `url`, returning the response body bytes.
    ///
    /// # Errors
    ///
    /// Returns an error on a transport failure or a non-success HTTP status.
    fn get(&self, url: &str, git_protocol: Option<&str>) -> Result<Vec<u8>>;

    /// Issue a `POST` to `url` with the given `content_type`, `accept` header,
    /// and request `body`, returning the response body bytes.
    ///
    /// # Errors
    ///
    /// Returns an error on a transport failure or a non-success HTTP status.
    fn post(
        &self,
        url: &str,
        content_type: &str,
        accept: &str,
        body: &[u8],
        git_protocol: Option<&str>,
    ) -> Result<Vec<u8>>;

    /// Issue a `GET` to `url` and return both the response body and the final
    /// URL the request resolved to after any HTTP redirects the client followed
    /// (`None` when the client does not track it).
    ///
    /// The smart-HTTP transport uses this on the `info/refs` discovery GET to
    /// re-base subsequent `git-upload-pack` POSTs onto a redirected location
    /// (Git's `http.followRedirects`): a host that redirects `info/refs` to a
    /// backing host expects the stateless-RPC POSTs there too, and many HTTP
    /// clients follow redirects on GET but not on POST. The default
    /// implementation calls [`get`](Self::get) and reports no final URL, so
    /// existing clients keep working unchanged (without redirect re-basing).
    ///
    /// # Errors
    ///
    /// Returns an error on a transport failure or a non-success HTTP status.
    fn get_with_final_url(
        &self,
        url: &str,
        git_protocol: Option<&str>,
    ) -> Result<(Vec<u8>, Option<String>)> {
        Ok((self.get(url, git_protocol)?, None))
    }

    /// The default `Git-Protocol` request-header value to apply when the caller
    /// passes `None`. Defaults to no header.
    fn git_protocol_header(&self) -> Option<&str> {
        None
    }

    /// Whether smart-HTTP is enabled (vs. dumb-HTTP fallback). Defaults to
    /// `true`; embedders that honor `GIT_SMART_HTTP=0` may return `false`.
    fn smart_http_enabled(&self) -> bool {
        true
    }

    /// Issue the same `POST` as [`post`](Self::post) but hand the body back as
    /// an [`HttpBody`] that yields it in pieces.
    ///
    /// The default buffers the whole response and yields it as a single chunk, so
    /// every existing client keeps working unchanged and pays exactly what it
    /// paid before. A client that can deliver a response incrementally -- a
    /// mobile `URLSession` delegate, for instance -- overrides this and stops
    /// needing the whole packfile in memory.
    ///
    /// # Errors
    ///
    /// Returns an error on a transport failure or a non-success HTTP status.
    fn post_streaming(
        &self,
        url: &str,
        content_type: &str,
        accept: &str,
        body: &[u8],
        git_protocol: Option<&str>,
    ) -> Result<Box<dyn HttpBody>> {
        let bytes = self.post(url, content_type, accept, body, git_protocol)?;
        Ok(Box::new(OneChunk {
            bytes: Some(bytes),
        }))
    }
}

/// Forward [`HttpClient`] through a shared [`std::sync::Arc`], so one client can
/// back several transports (and be observed by the caller) without moving it.
impl<C: HttpClient> HttpClient for std::sync::Arc<C> {
    fn get(&self, url: &str, git_protocol: Option<&str>) -> Result<Vec<u8>> {
        (**self).get(url, git_protocol)
    }

    fn post(
        &self,
        url: &str,
        content_type: &str,
        accept: &str,
        body: &[u8],
        git_protocol: Option<&str>,
    ) -> Result<Vec<u8>> {
        (**self).post(url, content_type, accept, body, git_protocol)
    }

    fn get_with_final_url(
        &self,
        url: &str,
        git_protocol: Option<&str>,
    ) -> Result<(Vec<u8>, Option<String>)> {
        (**self).get_with_final_url(url, git_protocol)
    }

    fn git_protocol_header(&self) -> Option<&str> {
        (**self).git_protocol_header()
    }

    fn smart_http_enabled(&self) -> bool {
        (**self).smart_http_enabled()
    }
}

/// A response body delivered in pieces rather than as one contiguous buffer.
///
/// `HttpClient::post` hands grit the whole `git-upload-pack` response at once, and the
/// packfile inside that response is the bulk of a fetch. A repository with a
/// few gigabytes of history therefore had to be resident in memory in full
/// before a single object was written -- once as the HTTP response, and again
/// as the de-framed pack. That is survivable on a desktop and fatal in a phone
/// process, which jetsam kills at a few hundred megabytes with no error at all.
///
/// This trait carries the same bytes split into pieces, so the parsing above it
/// can be written once against a pull-based source and driven by either.
pub trait HttpBody: Send {
    /// The next piece of the body, or `None` once it has ended.
    ///
    /// # Errors
    ///
    /// Returns an error if the body cannot be read further.
    fn next_chunk(&mut self) -> Result<Option<Vec<u8>>>;
}

/// The default `HttpClient::post_streaming` body: the whole response as one chunk.
struct OneChunk {
    bytes: Option<Vec<u8>>,
}

impl HttpBody for OneChunk {
    fn next_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        Ok(self.bytes.take())
    }
}

/// A pull-based `Read` over an `HttpBody`.
///
/// Everything downstream of the HTTP layer -- the pkt-line parser, the side-band
/// demuxer, `crate::unpack_objects` -- already works from a `Read`, so this is the
/// seam that lets a streamed body and a buffered one share all of it. With it,
/// `unpack_objects` can consume a pack of any size without that pack ever being
/// whole in memory.
/// The packfile a fetch is still delivering, or None when the remote sent
/// no pack at all.
///
/// Named so the fetch signatures need not spell out the lifetime that ties
/// the reader to the progress sink it reports side-band text into.
pub type PackStream = Option<SidebandPackReader>;
pub struct BodyReader {
    body: Box<dyn HttpBody>,
    current: VecDeque<u8>,
}

impl BodyReader {
    /// Wrap a body so it can be read as a `Read`.
    pub fn new(body: Box<dyn HttpBody>) -> Self {
        Self {
            body,
            current: VecDeque::new(),
        }
    }
}

impl std::io::Read for BodyReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while self.current.is_empty() {
            match self.body.next_chunk() {
                Ok(Some(chunk)) if !chunk.is_empty() => self.current.extend(chunk),
                Ok(Some(_)) => continue,
                Ok(None) => return Ok(0),
                Err(e) => return Err(std::io::Error::other(e.to_string())),
            }
        }
        let n = self.current.len().min(buf.len());
        for (slot, byte) in buf.iter_mut().take(n).zip(self.current.drain(..n)) {
            *slot = byte;
        }
        Ok(n)
    }
}

const UPLOAD_PACK: &str = "git-upload-pack";

/// Strip the optional `# service=...\n` pkt-line + flush preamble that a
/// smart-HTTP `info/refs?service=...` response begins with, returning the
/// remaining advertisement bytes.
///
/// A smart server prefixes the advertisement with `001e# service=git-upload-pack\n`
/// followed by a `0000` flush; a dumb server (or a raw `upload-pack
/// --advertise-refs` body) omits it. Lifted from the CLI's
/// `strip_v0_service_advertisement_if_present`.
fn strip_service_advertisement(body: &[u8]) -> Result<&[u8]> {
    let mut cur = Cursor::new(body);
    let start = cur.position();
    match pkt_line::read_packet(&mut cur)? {
        Some(pkt_line::Packet::Data(line)) if line.starts_with("# service=") => {
            // Consume the trailing flush after the service header.
            match pkt_line::read_packet(&mut cur)? {
                Some(pkt_line::Packet::Flush) | None => {}
                _ => {
                    // No flush after the service line: not a smart preamble; rewind.
                    return Ok(body);
                }
            }
            let pos = cur.position() as usize;
            Ok(&body[pos..])
        }
        _ => {
            cur.set_position(start);
            Ok(body)
        }
    }
}

/// A parsed v0/v1 advertisement ref entry (name -> oid).
#[derive(Clone, Debug)]
struct AdvRef {
    name: String,
    oid: ObjectId,
}

/// The discovery outcome: protocol version, advertised refs, capabilities, and
/// the symref target for `HEAD` (if any).
struct Discovery {
    protocol_version: u8,
    refs: Vec<AdvRef>,
    caps: HashSet<String>,
    head_symref: Option<String>,
    object_format: String,
}

/// Parse a v0/v1 ref advertisement (after the service preamble is stripped).
///
/// Hash-width aware via [`ObjectId::from_hex`]. Capabilities ride on the NUL
/// suffix of the first ref line; the `symref=HEAD:<target>` capability records
/// the default branch. The all-zero "unborn HEAD" carrier and `shallow`
/// trailers are skipped. Lifted from the CLI's `parse_v0_v1_advertisement` /
/// `discover_http_protocol`.
fn parse_advertisement(body: &[u8]) -> Result<Discovery> {
    let mut cur = Cursor::new(body);

    // Peek the first packet to distinguish v2 from v0/v1.
    let first = match pkt_line::read_packet(&mut cur)? {
        None | Some(pkt_line::Packet::Flush) => {
            // Empty advertisement (empty repo on an older server): no refs.
            return Ok(Discovery {
                protocol_version: 0,
                refs: Vec::new(),
                caps: HashSet::new(),
                head_symref: None,
                object_format: "sha1".to_owned(),
            });
        }
        Some(pkt_line::Packet::Data(s)) => s,
        Some(other) => {
            return Err(Error::Message(format!(
                "unexpected first advertisement packet: {other:?}"
            )))
        }
    };
    if first.trim_end() == "version 2" {
        // Detect v2 so the caller can report it as unsupported in this pass.
        let mut caps = HashSet::new();
        loop {
            match pkt_line::read_packet(&mut cur)? {
                None | Some(pkt_line::Packet::Flush) => break,
                Some(pkt_line::Packet::Data(s)) => {
                    caps.insert(s.trim_end().to_owned());
                }
                Some(_) => break,
            }
        }
        let object_format = caps
            .iter()
            .find_map(|c| c.strip_prefix("object-format="))
            .unwrap_or("sha1")
            .to_owned();
        return Ok(Discovery {
            protocol_version: 2,
            refs: Vec::new(),
            caps,
            head_symref: None,
            object_format,
        });
    }

    // v0/v1: rewind and parse the ref lines.
    cur.set_position(0);
    let mut refs = Vec::new();
    let mut caps: HashSet<String> = HashSet::new();
    let mut head_symref = None;
    let mut first_ref_line = true;
    loop {
        match pkt_line::read_packet(&mut cur)? {
            None | Some(pkt_line::Packet::Flush) => break,
            Some(pkt_line::Packet::Data(line)) => {
                let line = line.trim_end_matches('\n');
                if line.starts_with("version ") {
                    continue;
                }
                if line.starts_with("shallow ") || line.starts_with("unshallow ") {
                    continue;
                }
                let (payload, cap_part) = match line.split_once('\0') {
                    Some((p, c)) => (p.trim(), Some(c)),
                    None => (line.trim(), None),
                };
                let Some((oid_hex, refname)) =
                    payload.split_once('\t').or_else(|| payload.split_once(' '))
                else {
                    continue;
                };
                let oid_hex = oid_hex.trim();
                let refname = refname.trim();
                if first_ref_line {
                    if let Some(raw_caps) = cap_part {
                        for cap in raw_caps.split_whitespace() {
                            if let Some(target) = cap.strip_prefix("symref=HEAD:") {
                                head_symref = Some(target.to_owned());
                            }
                            caps.insert(cap.to_owned());
                        }
                    }
                    first_ref_line = false;
                }
                if refname.is_empty() {
                    continue;
                }
                // All-zero OID marks the unborn-HEAD capabilities carrier (empty repo).
                if oid_hex.bytes().all(|b| b == b'0') {
                    continue;
                }
                let oid = ObjectId::from_hex(oid_hex).map_err(|e| {
                    Error::Message(format!("bad oid in advertisement: {oid_hex}: {e}"))
                })?;
                refs.push(AdvRef {
                    name: refname.to_owned(),
                    oid,
                });
            }
            Some(other) => {
                return Err(Error::Message(format!(
                    "unexpected packet in advertisement: {other:?}"
                )))
            }
        }
    }
    let object_format = caps
        .iter()
        .find_map(|c| c.strip_prefix("object-format="))
        .unwrap_or("sha1")
        .to_owned();
    Ok(Discovery {
        protocol_version: if caps.contains("version 1") { 1 } else { 0 },
        refs,
        caps,
        head_symref,
        object_format,
    })
}

/// Build the `info/refs?service=git-upload-pack` discovery URL for `repo_url`.
fn info_refs_url(repo_url: &str) -> String {
    let base = repo_url.trim_end_matches('/');
    let mut url = format!("{base}/info/refs");
    url.push_str(if url.contains('?') { "&" } else { "?" });
    url.push_str("service=");
    url.push_str(UPLOAD_PACK);
    url
}

/// Given the `original_base` of an `info/refs` discovery request and the final
/// URL it resolved to after any HTTP redirects the client followed, return the
/// re-based repo URL to use for subsequent smart-HTTP requests — or `None` when
/// there was no usable redirect (keep the original base).
///
/// This implements the client side of Git's `http.followRedirects`: when a host
/// redirects `info/refs` to a backing location (e.g. tangled.org → its "knot"
/// host), the later `git-upload-pack` POSTs must target the redirected location.
/// Many HTTP clients follow the redirect on the discovery GET but not on a POST
/// (which then hits the redirecting host and comes back as an un-followed `3xx`
/// with an empty body — zero refs, a silently empty clone). The redirect must
/// preserve the `/info/refs` path suffix; the new base is the final URL with
/// that suffix (and any query) removed.
#[must_use]
pub fn rebased_base_from_redirect(original_base: &str, final_url: Option<&str>) -> Option<String> {
    let final_url = final_url?;
    let final_path = final_url.split('?').next().unwrap_or(final_url);
    let new_base = final_path.strip_suffix("/info/refs")?.trim_end_matches('/');
    if new_base.is_empty() || new_base == original_base.trim_end_matches('/') {
        return None;
    }
    Some(new_base.to_owned())
}

/// The `git-upload-pack` stateless-RPC endpoint URL for `repo_url`.
fn upload_pack_url(repo_url: &str) -> String {
    let base = repo_url.trim_end_matches('/');
    format!("{base}/{UPLOAD_PACK}")
}

/// A live smart-HTTP connection: the parsed advertisement plus the context
/// needed to issue the stateless-RPC POST. Smart HTTP is request/response, so
/// there is no persistent duplex socket — the `reader`/`writer` accessors are
/// not used by [`http_fetch`], which drives the POST loop directly.
///
/// `reader`/`writer` return empty/sink streams; embedders that want to drive a
/// custom negotiation should use [`http_fetch`] (or read the advertisement via
/// the accessors and POST through their [`HttpClient`]).
pub struct SmartHttpConnection {
    repo_url: String,
    adv_refs: Vec<(String, ObjectId)>,
    caps: Vec<String>,
    head_symref: Option<String>,
    protocol_version: u8,
    object_format: String,
    // Held so embedders/tests can identify which service this connection speaks.
    service: Service,
    empty_reader: Cursor<Vec<u8>>,
    sink: Vec<u8>,
}

impl SmartHttpConnection {
    /// The repository URL this connection targets.
    #[must_use]
    pub fn repo_url(&self) -> &str {
        &self.repo_url
    }

    /// The server's advertised object format (`sha1` or `sha256`).
    #[must_use]
    pub fn object_format(&self) -> &str {
        &self.object_format
    }

    /// The service this connection speaks.
    #[must_use]
    pub fn service(&self) -> Service {
        self.service
    }
}

impl Connection for SmartHttpConnection {
    fn reader(&mut self) -> &mut dyn Read {
        &mut self.empty_reader
    }

    fn writer(&mut self) -> &mut dyn Write {
        &mut self.sink
    }

    fn advertised_refs(&self) -> &[(String, ObjectId)] {
        &self.adv_refs
    }

    fn capabilities(&self) -> &[String] {
        &self.caps
    }

    fn head_symref(&self) -> Option<&str> {
        self.head_symref.as_deref()
    }

    fn protocol_version(&self) -> u8 {
        self.protocol_version
    }
}

/// A smart-HTTP [`Transport`] over a pluggable [`HttpClient`].
///
/// [`Transport::connect`] performs the `info/refs?service=git-upload-pack`
/// discovery GET and parses the advertisement; the returned [`Connection`]
/// exposes the advertised refs/capabilities. Use [`http_fetch`] to drive the
/// fetch negotiation over the same client.
pub struct SmartHttpTransport<C: HttpClient> {
    client: C,
}

impl<C: HttpClient> SmartHttpTransport<C> {
    /// Build a transport backed by `client`.
    pub fn new(client: C) -> Self {
        Self { client }
    }

    /// Borrow the underlying HTTP client.
    pub fn client(&self) -> &C {
        &self.client
    }

    /// Push `refs` to `repo_url` over smart HTTP (`git-receive-pack`), returning a
    /// [`crate::transfer::PushOutcome`].
    ///
    /// This is the push counterpart to [`http_fetch`]: it discovers the
    /// receive-pack advertisement, decides each update, builds the command block +
    /// pack, POSTs `git-receive-pack`, and parses the `report-status` reply —
    /// reusing the same decision/pack/report machinery as the duplex
    /// [`crate::push::push_remote`]. Delegates to [`crate::push::push_http`].
    ///
    /// Protocol v0/v1 only (a v2 receive-pack advertisement is rejected).
    ///
    /// # Errors
    ///
    /// Returns an error if discovery fails, the advertisement is protocol v2, a
    /// source object is missing locally, the pack build fails, or on wire/parse
    /// I/O failure.
    pub fn push(
        &self,
        local_git_dir: &Path,
        repo_url: &str,
        refs: &[crate::transfer::PushRefSpec],
        opts: &crate::transfer::PushOptions,
        progress: &mut dyn Progress,
    ) -> Result<crate::transfer::PushOutcome> {
        crate::push::push_http(&self.client, local_git_dir, repo_url, refs, opts, progress)
    }

    /// Perform the `info/refs` discovery for `repo_url` and `service`, returning
    /// the parsed [`Discovery`].
    ///
    /// `git_protocol` is the `Git-Protocol` request-header value to apply (e.g.
    /// `version=2` to request a v2 advertisement); when `None`, the client's
    /// default ([`HttpClient::git_protocol_header`]) is used.
    fn discover(
        &self,
        repo_url: &str,
        _service: Service,
        git_protocol: Option<&str>,
    ) -> Result<Discovery> {
        let url = info_refs_url(repo_url);
        let gp = git_protocol.or_else(|| self.client.git_protocol_header());
        let body = self.client.get(&url, gp)?;
        let stripped = strip_service_advertisement(&body)?;
        parse_advertisement(stripped)
    }
}

/// The `Git-Protocol` request-header value for a requested protocol version, or
/// `None` for v0 (no header — the classic advertisement).
fn git_protocol_for_version(version: u8) -> Option<String> {
    if version >= 1 {
        Some(format!("version={version}"))
    } else {
        None
    }
}

impl<C: HttpClient> Transport for SmartHttpTransport<C> {
    fn connect(
        &self,
        url: &str,
        service: Service,
        opts: &ConnectOptions,
    ) -> Result<Box<dyn Connection>> {
        // Request the protocol version the caller asked for via the
        // `Git-Protocol` header (a v2 server only returns its v2 capability
        // advertisement when it sees `version=2`); fall back to the client's
        // default header otherwise. The server may still downgrade.
        crate::net_trace::net_trace!(
            "http(s) discover {url} (service={}, request protocol v{})",
            service.wire_name(),
            opts.protocol_version
        );
        let gp = git_protocol_for_version(opts.protocol_version);
        let disc = self.discover(url, service, gp.as_deref())?;
        let adv_refs: Vec<(String, ObjectId)> = disc
            .refs
            .iter()
            .filter(|r| r.name != "HEAD" && !r.name.ends_with("^{}"))
            .map(|r| (r.name.clone(), r.oid))
            .collect();
        let caps: Vec<String> = disc.caps.iter().cloned().collect();
        crate::net_trace::net_trace!(
            "http(s) discovered: protocol v{}, {} ref(s) advertised",
            disc.protocol_version,
            adv_refs.len()
        );
        Ok(Box::new(SmartHttpConnection {
            repo_url: url.to_owned(),
            adv_refs,
            caps,
            head_symref: disc.head_symref,
            protocol_version: disc.protocol_version,
            object_format: disc.object_format,
            service,
            empty_reader: Cursor::new(Vec::new()),
            sink: Vec::new(),
        }))
    }
}

/// Read a length-prefixed pkt-line payload, returning `None` on flush/delim/EOF.
fn read_pkt_payload(r: &mut impl Read) -> std::io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len_str = std::str::from_utf8(&len_buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let len = usize::from_str_radix(len_str, 16)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    match len {
        0..=2 => Ok(None),
        n if n <= 4 => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid pkt-line length: {n}"),
        )),
        n => {
            let mut buf = vec![0u8; n - 4];
            r.read_exact(&mut buf)?;
            Ok(Some(buf))
        }
    }
}

/// Kind of `ACK` status suffix in a v0 negotiation response.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AckKind {
    /// `ACK <oid>` with no status suffix (ends a round / post-`done`).
    Bare,
    /// `ACK <oid> common` — the server holds this commit; replay it on the next
    /// stateless RPC if we had not already marked it common.
    Common,
    /// `ACK <oid> continue` — recorded in the negotiator but not replayed.
    Continue,
    /// `ACK <oid> ready` — the server has enough; it will send the pack.
    Ready,
}

struct Ack {
    oid: ObjectId,
    kind: AckKind,
}

fn parse_ack(line: &str) -> Option<Ack> {
    let rest = line.strip_prefix("ACK ")?;
    let hex = rest.split_whitespace().next()?;
    let oid = ObjectId::from_hex(hex).ok()?;
    let tail = rest.strip_prefix(hex).unwrap_or("").trim();
    let kind = if tail.contains("continue") {
        AckKind::Continue
    } else if tail.contains("common") {
        AckKind::Common
    } else if tail.contains("ready") {
        AckKind::Ready
    } else {
        AckKind::Bare
    };
    Some(Ack { oid, kind })
}

/// Result of parsing one stateless-RPC response.
struct RoundResult {
    acks: Vec<Ack>,
    got_pack: bool,
    /// Shallow boundaries the server reported (`shallow <oid>`) in this response's
    /// leading `shallow-info` section (empty unless a deepen was requested).
    shallow: Vec<ObjectId>,
    /// Boundaries the server un-shallowed (`unshallow <oid>`) in this response.
    unshallow: Vec<ObjectId>,
}

/// Demultiplex the side-band pack from a stateless-RPC response, appending pack
/// bytes to `out` and forwarding channel-2 progress. Mirrors the CLI's
/// `read_sideband_pack_until_done`.
pub struct SidebandPackReader {
    src: Box<dyn Read>,
    /// Bytes to serve before touching `src` again, after a rewind.
    pushback: Vec<u8>,
    /// Everything read from `src` since the last mark, so a rewind can replay it.
    recorded: Vec<u8>,
    /// Emit only side-band channel 1. When false the stream is a bare packfile.
    sideband: bool,
    /// True until the control section has been consumed. Reading the pack before
    /// that would swallow the ACK/NAK lines that precede it.
    in_control: bool,
    /// Decoded pack bytes not yet handed to the consumer.
    ready: VecDeque<u8>,
    /// Held back while the `PACK` magic is located across a packet boundary.
    held: Vec<u8>,
    seen_pack: bool,
    finished: bool,
    /// Where side-band progress text goes, non-null only while a pack is
    /// being read.
    ///
    /// A raw pointer rather than a reference on purpose. A reference would
    /// pin the borrow of the progress sink for the whole fetch, but the sink
    /// is only live for the unpack that reads from this reader.
    ///
    /// SAFETY: unpack_into sets it immediately before the unpack and clears it
    /// immediately after, and nothing else writes to it, so it cannot outlive
    /// the progress reference handed to that call.
    progress: Option<*mut (dyn Progress + 'static)>,
}

impl SidebandPackReader {
    /// Take a response body and prepare to read the packfile out of it.
    pub fn new(src: Box<dyn Read>, sideband: bool) -> Self {
        Self {
            src,
            pushback: Vec::new(),
            recorded: Vec::new(),
            sideband,
            in_control: true,
            ready: VecDeque::new(),
            held: Vec::new(),
            seen_pack: false,
            finished: false,
            progress: None,
        }
    }

    /// Forget what has been read so far, so the next read re-reads it.
    ///
    /// Used by the control parser: it reads a packet to see what it is, and has
    /// to give it back when the answer turns out to be "not mine".
    pub fn mark(&mut self) {
        self.recorded.clear();
    }

    /// Replay everything read since the last [`mark`](Self::mark).
    pub fn rewind(&mut self) {
        if self.recorded.is_empty() {
            return;
        }
        let mut replay = std::mem::take(&mut self.recorded);
        replay.extend_from_slice(&self.pushback);
        self.pushback = replay;
    }

    /// One pkt-line payload from the control section, or `None` at a flush
    /// or the end of the response.
    pub fn read_control_payload(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        read_pkt_payload(&mut ControlReader(self))
    }

    /// A reader over a body that has already been collected in full.
    ///
    /// The protocol-v2 path still de-frames into a Vec through the shared
    /// fetch.rs helper, so it arrives here whole. It gets the same reader type
    /// so both paths converge on one ingest; nothing about v2 streams yet.
    pub fn from_bytes(bytes: Vec<u8>, sideband: bool) -> Self {
        let reader = std::io::Cursor::new(bytes);
        Self {
            src: Box::new(reader),
            pushback: Vec::new(),
            recorded: Vec::new(),
            sideband,
            in_control: false,
            ready: VecDeque::new(),
            held: Vec::new(),
            seen_pack: false,
            finished: false,
            progress: None,
        }
    }
    /// Validate the packfile signature, then unpack straight into the odb as the
    /// bytes arrive.
    ///
    /// Progress is taken here rather than held from construction so its borrow
    /// lasts exactly as long as the unpack, instead of pinning the sink for the
    /// whole fetch. not_a_pack is what to report when the stream does not start
    /// with the magic, which is the job the old length-and-prefix check did.
    pub fn unpack_into(
        &mut self,
        odb: &crate::odb::Odb,
        opts: &crate::unpack_objects::UnpackOptions,
        progress: &mut dyn Progress,
        not_a_pack: &str,
    ) -> Result<usize> {
        self.expect_pack_magic(not_a_pack)?;
        // SAFETY: the erased lifetime is sound because the sink is dropped
        // again before this function returns, so it cannot outlive the
        // reference it was taken from.
        self.progress = Some(unsafe {
            std::mem::transmute::<*mut dyn Progress, *mut (dyn Progress + 'static)>(
                std::ptr::from_mut(progress),
            )
        });
        let result = crate::unpack_objects::unpack_objects(self, odb, opts);
        self.progress = None;
        result
    }
    /// One pkt-line from the control section, with the same distinctions
    /// grit's parser makes. None means the response ended.
    pub fn read_control_packet(&mut self) -> std::io::Result<Option<pkt_line::Packet>> {
        pkt_line::read_packet(&mut ControlReader(self))
    }

    /// Check the packfile signature without consuming it.
    ///
    /// The old code took a length-and-prefix slice check on a body that was
    /// already whole. With a streamed body the magic has to be inspected where
    /// it lands, and it must be left there: the pack stream is served from
    /// [Self::ready], so putting the bytes back anywhere else would replay them
    /// in the middle of the pack and shift every object after it.
    pub fn expect_pack_magic(&mut self, not_a_pack: &str) -> Result<()> {
        while self.ready.len() < 4 {
            if self.finished {
                break;
            }
            self.fill()?;
        }
        // A VecDeque is not contiguous, so the magic is compared a byte at a
        // time rather than sliced: &self.ready[..4] parses as a second index
        // into the first element, not as a range over the queue.
        let magic = self.ready.iter().take(4).copied().collect::<Vec<u8>>();
        if magic.as_slice() != b"PACK" {
            return Err(Error::Message(not_a_pack.to_owned()));
        }
        Ok(())
    }
    /// Hand the stream over to the packfile consumer: from here on, reads
    /// come back as packfile bytes rather than as control lines.

    pub fn begin_pack(&mut self) {
        self.in_control = false;
    }

    /// Read from the pushback buffer first, then the body, recording what the
    /// body produced so a rewind can replay it.
    fn read_from_src(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if !self.pushback.is_empty() {
            let n = self.pushback.len().min(buf.len());
            buf[..n].copy_from_slice(&self.pushback[..n]);
            self.pushback.drain(..n);
            return Ok(n);
        }
        let n = self.src.read(buf)?;
        if self.in_control {
            self.recorded.extend_from_slice(&buf[..n]);
        }
        Ok(n)
    }


    /// Bytes the reader is holding on behalf of a consumer, for tests.
    ///
    /// A reader that buffered instead of streaming would grow this to the
    /// size of the packfile. That is the whole claim of this type, so it is
    /// measured rather than argued.
    #[cfg(test)]
    pub fn retained_bytes(&self) -> usize {
        self.ready.len() + self.held.len() + self.pushback.len()
    }
    /// Pull side-band packets until at least one pack byte is decoded.
    fn fill(&mut self) -> std::io::Result<()> {
        loop {
            let Some(payload) = self.read_control_payload()? else {
                self.finished = true;
                return Ok(());
            };
            if payload.is_empty() {
                continue;
            }
            match payload[0] {
                1 => append_pack_data(
                    &payload[1..],
                    &mut self.ready,
                    &mut self.held,
                    &mut self.seen_pack,
                ),
                2 => {
                    if let Some(sink) = self.progress {
                        // SAFETY: set and cleared by unpack_into.
                        unsafe { (*sink).message(&payload[1..]) };
                    }
                }
                3 => {
                    return Err(std::io::Error::other(format!(
                        "remote error: {}",
                        String::from_utf8_lossy(&payload[1..]).trim_end()
                    )))
                }
                _ => append_pack_data(
                    &payload,
                    &mut self.ready,
                    &mut self.held,
                    &mut self.seen_pack,
                ),
            }
            if !self.ready.is_empty() {
                return Ok(());
            }
        }
    }
}

/// A [Read] view of the control section, for handing to grit's pkt-line parser.
///
/// [SidebandPackReader]'s own [Read] impl is deliberately closed while the
/// control section is being parsed, so that nothing can make the demuxer
/// swallow the ACK/NAK lines that precede the pack. This is the way in for a
/// parser, and it routes through the same bookkeeping that [mark](Self::mark)
/// and [rewind](Self::rewind) depend on.
struct ControlReader<'a>(&'a mut SidebandPackReader);

impl Read for ControlReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read_from_src(buf)
    }
}

impl Read for SidebandPackReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.in_control {
            return Err(std::io::Error::other(
                "the control section must be consumed before reading the packfile"
            ));
        }
        if !self.sideband {
            // A bare packfile needs no de-framing; hand the body straight on.
            return self.read_from_src(buf);
        }
        while self.ready.is_empty() {
            if self.finished {
                return Ok(0);
            }
            self.fill()?;
        }
        let n = self.ready.len().min(buf.len());

        for (slot, byte) in buf.iter_mut().take(n).zip(self.ready.drain(..n)) {
            *slot = byte;
        }
        Ok(n)
    }
}

/// Append channel-1 (or raw) data to `out`, scanning for the `PACK` magic that
/// may straddle chunk boundaries.
fn append_pack_data(
    data: &[u8],
    out: &mut VecDeque<u8>,
    pending: &mut Vec<u8>,
    seen_pack: &mut bool,
) {
    if *seen_pack {
        out.extend(data.iter().copied());
        return;
    }
    pending.extend_from_slice(data);
    if let Some(pos) = pending.windows(4).position(|w| w == b"PACK") {
        *seen_pack = true;
        out.extend(pending[pos..].iter().copied());
        pending.clear();
    } else if pending.len() > 3 {
        let keep_from = pending.len() - 3;
        pending.drain(..keep_from);
    }
}

/// Parse one v0 stateless-RPC `git-upload-pack` response: an optional leading
/// `shallow-info` section (only when `expect_shallow`, i.e. a deepen was
/// requested), then optional `ACK`/`NAK` negotiation lines, then (if the server
/// is generating one) the side-band pack.
///
/// On return `src` is handed over to the packfile: it has either been
/// positioned at the pack, or the response held no pack at all. The pack is not
/// collected here, because the entire point is that it never has to be.
fn read_stateless_response(
    src: &mut SidebandPackReader,
    sideband: bool,
    expect_shallow: bool,
) -> Result<RoundResult> {
    let mut acks = Vec::new();
    let mut shallow = Vec::new();
    let mut unshallow = Vec::new();
    let mut got_pack = false;

    // Shallow-info section: `shallow`/`unshallow` lines terminated by a flush. A
    // server with nothing to report still emits the trailing flush. Rewind and
    // fall through if the first line is not a shallow-info line (no section).
    if expect_shallow {
        loop {
            src.mark();
            match src.read_control_packet()? {
                None | Some(pkt_line::Packet::Flush) => break,
                Some(pkt_line::Packet::Data(line)) => {
                    if let Some(rest) = line.strip_prefix("shallow ") {
                        if let Ok(oid) = ObjectId::from_hex(rest.trim()) {
                            shallow.push(oid);
                        }
                    } else if let Some(rest) = line.strip_prefix("unshallow ") {
                        if let Ok(oid) = ObjectId::from_hex(rest.trim()) {
                            unshallow.push(oid);
                        }
                    } else {
                        src.rewind();
                        break;
                    }
                }
                Some(_) => break,
            }
        }
    }

    loop {
        src.mark();
        let Some(payload) = src.read_control_payload()? else {
            break;
        };
        if payload.is_empty() {
            continue;
        }
        let is_pack =
            (sideband && payload.first() == Some(&1) && payload.get(1..5) == Some(b"PACK"))
                || payload.starts_with(b"PACK");
        if is_pack {
            // Give the packet back so the demuxer re-reads it as the start of
            // the pack, rather than the parser and the pack reader each taking
            // a guess at where it begins.
            src.rewind();
            src.begin_pack();
            got_pack = true;
            break;
        }
        let text = String::from_utf8_lossy(&payload);
        let line = text.trim_end_matches('\n');
        if let Some(err) = line.strip_prefix("ERR ") {
            return Err(Error::Message(format!("remote upload-pack error: {err}")));
        }
        if line == "NAK" {
            continue;
        }
        if let Some(ack) = parse_ack(line) {
            acks.push(ack);
        }
    }
    Ok(RoundResult {
        acks,
        got_pack,
        shallow,
        unshallow,
    })
}

/// The v0/v1 fetch capabilities we request, intersected with what the server
/// advertised. Mirrors `build_fetch_caps_v0`.
fn build_fetch_caps(caps: &HashSet<String>) -> String {
    let mut enabled = Vec::new();
    let multi_ack_detailed = caps.contains("multi_ack_detailed");
    if multi_ack_detailed {
        enabled.push("multi_ack_detailed");
    }
    if multi_ack_detailed && caps.contains("no-done") {
        enabled.push("no-done");
    }
    for want in [
        "side-band-64k",
        "thin-pack",
        "no-progress",
        "include-tag",
        "ofs-delta",
    ] {
        if caps.contains(want) {
            enabled.push(want);
        }
    }
    if enabled.is_empty() {
        String::new()
    } else {
        format!(" {}", enabled.join(" "))
    }
}

/// Next stateless-RPC `have` batch size (mirrors `fetch-pack.c` `next_flush`).
fn next_flush(count: usize) -> usize {
    const LARGE_FLUSH: usize = 16384;
    if count < LARGE_FLUSH {
        count * 2
    } else {
        count * 11 / 10
    }
}

/// Append the v0/v1 shallow/deepen request lines (the client's `shallow <oid>`
/// grafts and any `deepen`/`deepen-since`/`deepen-not`) to the persistent request
/// `state`, gated on the server capability where one exists. Mirrors the CLI's
/// `append_fetch_request_extensions_v0_v1`.
fn append_shallow_request_v0_http(
    req: &mut Vec<u8>,
    caps: &HashSet<String>,
    local_shallow: &[ObjectId],
    opts: &FetchOptions,
) -> Result<()> {
    for oid in local_shallow {
        pkt_line::write_line_to_vec(req, &format!("shallow {}", oid.to_hex()))?;
    }
    if opts.unshallow {
        pkt_line::write_line_to_vec(req, &format!("deepen {}", crate::shallow::INFINITE_DEPTH))?;
    } else if let Some(depth) = opts.depth.filter(|d| *d > 0) {
        pkt_line::write_line_to_vec(req, &format!("deepen {depth}"))?;
    }
    if let Some(since) = opts
        .deepen_since
        .as_deref()
        .filter(|s| !s.trim().is_empty())
    {
        if caps.contains("deepen-since") {
            let value = crate::shallow::deepen_since_wire_value(since);
            pkt_line::write_line_to_vec(req, &format!("deepen-since {value}"))?;
        }
    }
    if caps.contains("deepen-not") {
        for excl in &opts.deepen_not {
            let excl = excl.trim();
            if !excl.is_empty() {
                pkt_line::write_line_to_vec(req, &format!("deepen-not {excl}"))?;
            }
        }
    }
    Ok(())
}

/// Wrap a body that was collected whole into a reader.
///
/// Protocol v2 still de-frames into a Vec through the shared fetch.rs helper,
/// so this is the one place where the pack is fully resident. The reader type
/// is the same so both protocols converge on one ingest path, and so a future
/// streaming v2 is a change of where the bytes come from, not of what consumes
/// them.
fn pack_stream(bytes: Vec<u8>, sideband: bool) -> PackStream {
    if bytes.is_empty() {
        None
    } else {
        Some(SidebandPackReader::from_bytes(bytes, sideband))
    }
}
/// Negotiate and download the pack for `wants` over stateless-RPC HTTP,
/// returning a reader over the pack as it arrives (None if the server sent
/// none) plus any shallow-boundary updates the server reported.
fn negotiate_pack_http(
    client: &dyn HttpClient,
    local_git_dir: &Path,
    repo_url: &str,
    caps: &HashSet<String>,
    advertised: &[AdvRef],
    wants: &[ObjectId],
    opts: &FetchOptions,
    local_shallow: &[ObjectId],
) -> Result<(PackStream, crate::fetch::ShallowUpdate)> {
    let post_url = upload_pack_url(repo_url);
    let content_type = format!("application/x-{UPLOAD_PACK}-request");
    let accept = format!("application/x-{UPLOAD_PACK}-result");
    let fetch_caps = build_fetch_caps(caps);
    let sideband = caps.contains("side-band-64k");
    let multi_ack_detailed = caps.contains("multi_ack_detailed");
    let no_done = multi_ack_detailed && caps.contains("no-done");

    // A deepen/shallow request precedes the pack with a `shallow-info` section and
    // does not offer local haves (its objects bottom out at grafts).
    let shallow_request = opts.has_deepen_request() || !local_shallow.is_empty();

    let want_set: HashSet<ObjectId> = wants.iter().copied().collect();

    // Build the persistent request prefix replayed on every RPC: the want lines
    // (capabilities on the first), the shallow/deepen extensions, and the
    // terminating flush.
    let mut state = Vec::new();
    let first = wants[0];
    pkt_line::write_line_to_vec(
        &mut state,
        &format!("want {}{}", first.to_hex(), fetch_caps),
    )?;
    for w in wants.iter().skip(1) {
        pkt_line::write_line_to_vec(&mut state, &format!("want {}", w.to_hex()))?;
    }
    append_shallow_request_v0_http(&mut state, caps, local_shallow, opts)?;
    pkt_line::write_flush(&mut state)?;

    let mut shallow_update = crate::fetch::ShallowUpdate::default();

    // Build the negotiator from local tips, marking advertised tips we already
    // have as known-common. Skipped for a shallow request.
    let local_repo = crate::repo::Repository::open(local_git_dir, None)?;
    let mut negotiator = SkippingNegotiator::new(local_repo);
    if !shallow_request {
        for w in wants {
            if negotiator.repo().odb.read(w).is_ok() {
                negotiator.add_tip(*w)?;
            }
        }
        let mut tips: Vec<ObjectId> = Vec::new();
        for prefix in ["refs/heads/", "refs/tags/"] {
            if let Ok(entries) = crate::refs::list_refs(local_git_dir, prefix) {
                for (_, oid) in entries {
                    if negotiator.repo().odb.read(&oid).is_ok() {
                        tips.push(oid);
                    }
                }
            }
        }
        if let Ok(h) = crate::refs::resolve_ref(local_git_dir, "HEAD") {
            if negotiator.repo().odb.read(&h).is_ok() {
                tips.push(h);
            }
        }
        tips.sort_by_key(ObjectId::to_hex);
        tips.dedup();
        for t in tips {
            if want_set.contains(&t) {
                continue;
            }
            negotiator.add_tip(t)?;
        }
        for e in advertised {
            if want_set.contains(&e.oid) {
                continue;
            }
            if negotiator.repo().odb.read(&e.oid).is_ok() {
                negotiator.known_common(e.oid)?;
            }
        }
    }

    // The pack is returned as a reader rather than accumulated here: the caller
    // unpacks it as it arrives, which is what keeps a large fetch out of memory.
    let mut got_ready = false;
    let mut shallow_applied = false;

    const INITIAL_FLUSH: usize = 16;
    let mut count: usize = 0;
    let mut flush_at: usize = INITIAL_FLUSH;
    let mut round = Vec::new();
    // The negotiator is empty for a shallow request, so this loop is skipped and
    // the single `done` RPC below carries the wants + shallow lines.
    while let Some(oid) = negotiator.next_have()? {
        pkt_line::write_line_to_vec(&mut round, &format!("have {}", oid.to_hex()))?;
        count += 1;
        if count < flush_at {
            continue;
        }
        flush_at = next_flush(count);

        let mut req = state.clone();
        req.extend_from_slice(&round);
        pkt_line::write_flush(&mut req)?;
        round.clear();

        let body = client.post_streaming(&post_url, &content_type, &accept, &req, None)?;
        let mut pack =
            SidebandPackReader::new(Box::new(BodyReader::new(body)), sideband);
        let round_result = read_stateless_response(&mut pack, sideband, shallow_request)?;
        if shallow_request && !shallow_applied {
            shallow_update
                .shallow
                .extend(round_result.shallow.iter().copied());
            shallow_update
                .unshallow
                .extend(round_result.unshallow.iter().copied());
            shallow_applied = true;
        }
        for ack in &round_result.acks {
            if matches!(ack.kind, AckKind::Bare) {
                continue;
            }
            let was_common = negotiator.ack(ack.oid)?;
            if matches!(ack.kind, AckKind::Common) && !was_common {
                pkt_line::write_line_to_vec(&mut state, &format!("have {}", ack.oid.to_hex()))?;
            }
            if matches!(ack.kind, AckKind::Ready) {
                got_ready = true;
            }
        }
        if round_result.got_pack {
            return Ok((Some(pack), shallow_update));
        }
        if got_ready {
            break;
        }
    }

    // Final RPC ending in `done`, unless the pack already arrived with
    // `ACK ... ready` under `no-done`.
    if !(got_ready && no_done) {
        let mut req = state.clone();
        pkt_line::write_line_to_vec(&mut req, "done")?;
        pkt_line::write_flush(&mut req)?;
        let body = client.post_streaming(&post_url, &content_type, &accept, &req, None)?;
        let mut pack =
            SidebandPackReader::new(Box::new(BodyReader::new(body)), sideband);
        let round_result = read_stateless_response(&mut pack, sideband, shallow_request)?;
        if shallow_request && !shallow_applied {
            shallow_update.shallow.extend(round_result.shallow);
            shallow_update.unshallow.extend(round_result.unshallow);
        }
        if round_result.got_pack {
            return Ok((Some(pack), shallow_update));
        }
    }

    Ok((None, shallow_update))
}

/// Resolve the `wants` for a fetch from the advertised refs and the matched set.
///
/// Returns the matched ref records (for later ref-update classification) and the
/// set of wanted oids.
struct MatchPlan {
    matched: Vec<crate::transfer::MatchedRef>,
    wants: HashSet<ObjectId>,
    seen: HashSet<String>,
}

fn match_refspecs(
    remote_refs: &[(String, ObjectId)],
    positive: &[RefspecItem],
    negatives: &[RefspecItem],
) -> MatchPlan {
    let mut matched: Vec<crate::transfer::MatchedRef> = Vec::new();
    let mut wants: HashSet<ObjectId> = HashSet::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (name, oid) in remote_refs {
        if ref_excluded(name, negatives) {
            continue;
        }
        if let Some(local_ref) = match_positive(name, positive) {
            if seen.insert(name.clone()) {
                wants.insert(*oid);
                matched.push(crate::transfer::MatchedRef {
                    remote_ref: name.clone(),
                    local_ref,
                    oid: *oid,
                    force: refspecs_force(name, positive),
                    is_tag: name.starts_with("refs/tags/"),
                });
            }
        }
    }
    MatchPlan {
        matched,
        wants,
        seen,
    }
}

/// Fetch from a smart-HTTP remote, driving the stateless-RPC negotiation and
/// writing tracking-ref updates into `local_git_dir`.
///
/// This is the HTTP counterpart to [`crate::fetch::fetch_remote`]: instead of a
/// duplex socket it issues `info/refs` discovery + `git-upload-pack` POSTs
/// through `client`. The refspec matching, tag-mode, prune, and update
/// classification reuse the shared [`crate::transfer`] helpers, so the
/// [`FetchOutcome`] shape matches every other fetch path.
///
/// Both protocol v0/v1 and protocol v2 are handled: the version is taken from
/// the `info/refs` advertisement (the v2 capability block is returned only when
/// the discovery GET carries `Git-Protocol: version=2`, which the client's
/// default header supplies). For v2 the ref map is recovered with a
/// `command=ls-refs` POST and the pack is negotiated with `command=fetch` POSTs
/// (stateless: every round resends the wants + accumulated haves).
///
/// # Errors
///
/// Returns an error if discovery fails, a refspec is invalid, or negotiation /
/// pack ingest / ref I/O fails.
pub fn http_fetch(
    client: &dyn HttpClient,
    local_git_dir: &Path,
    repo_url: &str,
    opts: &FetchOptions,
    progress: &mut dyn Progress,
) -> Result<FetchOutcome> {
    use crate::net_trace::net_trace;
    net_trace!(
        "http_fetch: begin — {} ({} refspec(s), tags={:?})",
        repo_url,
        opts.refspecs.len(),
        opts.tags
    );
    // 1. Discovery (request v2 via the client's default `Git-Protocol` header;
    // a v0/v1 server ignores it and returns the classic advertisement). If the
    // server redirects `info/refs` to another location, re-base every following
    // request onto it (Git's `http.followRedirects`): re-fetch discovery from
    // the new base so the request carries the `?service=` query a redirect may
    // drop, and so the stateless-RPC POSTs target the redirected host (which a
    // client that won't follow a POST redirect would otherwise miss).
    let info_url = info_refs_url(repo_url);
    let (body, final_url) = client.get_with_final_url(&info_url, client.git_protocol_header())?;
    let rebased = rebased_base_from_redirect(repo_url, final_url.as_deref());
    let repo_url_owned;
    let (repo_url, disc) = match rebased {
        Some(new_base) => {
            net_trace!("http_fetch: redirected base {repo_url} -> {new_base}");
            let url = info_refs_url(&new_base);
            let body = client.get(&url, client.git_protocol_header())?;
            let disc = parse_advertisement(strip_service_advertisement(&body)?)?;
            repo_url_owned = new_base;
            (repo_url_owned.as_str(), disc)
        }
        None => {
            let disc = parse_advertisement(strip_service_advertisement(&body)?)?;
            (repo_url, disc)
        }
    };
    net_trace!(
        "http_fetch: discovered protocol v{}, {} ref(s)",
        disc.protocol_version,
        disc.refs.len()
    );
    if disc.protocol_version >= 2 {
        net_trace!("http_fetch: delegating to v2 stateless fetch");
        return http_fetch_v2(client, local_git_dir, repo_url, &disc, opts, progress);
    }

    let local_odb = open_odb(local_git_dir);

    let default_branch = disc
        .head_symref
        .as_deref()
        .map(|t| t.strip_prefix("refs/heads/").unwrap_or(t).to_owned());

    let remote_refs: Vec<(String, ObjectId)> = disc
        .refs
        .iter()
        .filter(|r| r.name != "HEAD" && !r.name.ends_with("^{}"))
        .map(|r| (r.name.clone(), r.oid))
        .collect();

    // 2. Parse refspecs.
    let mut positive: Vec<RefspecItem> = Vec::new();
    let mut negatives: Vec<RefspecItem> = Vec::new();
    for spec in &opts.refspecs {
        let item = parse_fetch_refspec(spec)
            .map_err(|e| Error::Message(format!("invalid refspec '{spec}': {e}")))?;
        if item.negative {
            negatives.push(item);
        } else {
            positive.push(item);
        }
    }
    for spec in &opts.negative_refspecs {
        let item = parse_fetch_refspec(spec)
            .map_err(|e| Error::Message(format!("invalid negative refspec '{spec}': {e}")))?;
        negatives.push(item);
    }

    // 3. Match refs to refspecs.
    let MatchPlan {
        mut matched,
        mut wants,
        mut seen,
    } = match_refspecs(&remote_refs, &positive, &negatives);

    // 4. TagMode: add tags (the wire `include-tag` capability brings tag
    // objects with the pack; All adds every advertised tag, Following adds them
    // provisionally and prunes unreachable ones after the pack lands).
    if opts.tags != TagMode::None {
        for (name, oid) in &remote_refs {
            if !name.starts_with("refs/tags/") {
                continue;
            }
            if seen.contains(name) || ref_excluded(name, &negatives) {
                continue;
            }
            seen.insert(name.clone());
            wants.insert(*oid);
            matched.push(crate::transfer::MatchedRef {
                remote_ref: name.clone(),
                local_ref: Some(name.clone()),
                oid: *oid,
                force: false,
                is_tag: true,
            });
        }
    }

    // 5. Wants → negotiate + ingest the pack. Normally the matched oids absent
    // locally; for a deepen/`--unshallow` request we must still `want` the tips
    // even if present so the server fills in ancestors past the old boundary.
    let local_shallow = crate::shallow::load_shallow_oids(local_git_dir)?;
    let shallow_request = opts.has_deepen_request() || !local_shallow.is_empty();
    let need: Vec<ObjectId> = if shallow_request {
        wants.iter().copied().collect()
    } else {
        wants
            .iter()
            .copied()
            .filter(|oid| !local_odb.exists(oid))
            .collect()
    };

    let mut shallow_update = crate::fetch::ShallowUpdate::default();

    if !need.is_empty() && !opts.dry_run {
        let (pack, su) = negotiate_pack_http(
            client,
            local_git_dir,
            repo_url,
            &disc.caps,
            &disc.refs,
            &need,
            opts,
            &local_shallow,
        )?;
        shallow_update = su;
        if let Some(mut pack) = pack {
            pack.unpack_into(
                &local_odb,
                &crate::unpack_objects::UnpackOptions {
                    quiet: true,
                    ..Default::default()
                },
                progress,
                "did not receive a valid pack from HTTP fetch",
            )?;
        }
    }

    // Apply shallow/unshallow boundary updates to the on-disk `shallow` file.
    if !opts.dry_run {
        crate::shallow::apply_shallow_updates(
            local_git_dir,
            &shallow_update.shallow,
            &shallow_update.unshallow,
        )?;
    }

    // 6. For TagMode::Following, drop tags whose target did not arrive.
    if opts.tags == TagMode::Following {
        retain_following_tags(&local_odb, &mut matched, &wants);
    }

    // 7. Classify + apply ref updates.
    let local_repo = if opts.dry_run {
        None
    } else {
        crate::repo::Repository::open(local_git_dir, None).ok()
    };

    let mut updates: Vec<RefUpdate> = Vec::new();
    if opts.prune {
        prune_tracking_refs(
            local_git_dir,
            &positive,
            &remote_refs,
            opts.dry_run,
            &mut updates,
        )?;
    }

    for m in &matched {
        let Some(local_ref) = &m.local_ref else {
            updates.push(RefUpdate {
                remote_ref: m.remote_ref.clone(),
                local_ref: None,
                old_oid: None,
                new_oid: Some(m.oid),
                mode: UpdateMode::NoChangeNeeded,
                note: Some("not stored (empty destination)".to_owned()),
            });
            continue;
        };
        let old = crate::refs::resolve_ref(local_git_dir, local_ref).ok();
        let mode = classify_update(old.as_ref(), &m.oid, m.force, m.is_tag, local_repo.as_ref());
        let write = matches!(
            mode,
            UpdateMode::New | UpdateMode::FastForward | UpdateMode::Forced
        );
        if write && !opts.dry_run {
            crate::refs::write_ref(local_git_dir, local_ref, &m.oid)?;
        }
        updates.push(RefUpdate {
            remote_ref: m.remote_ref.clone(),
            local_ref: Some(local_ref.clone()),
            old_oid: old,
            new_oid: Some(m.oid),
            mode,
            note: None,
        });
    }

    net_trace!("http_fetch: done — {} ref update(s)", updates.len());
    Ok(FetchOutcome {
        updates,
        default_branch,
        new_shallow: shallow_update.shallow,
        new_unshallow: shallow_update.unshallow,
    })
}

/// Fetch from a smart-HTTP remote that speaks protocol v2 (stateless multi-POST).
///
/// `disc` is the already-parsed v2 capability advertisement (no refs). This
/// recovers the ref map with a `command=ls-refs` POST, matches refspecs / tags
/// with the same shared [`crate::transfer`] helpers as the v0/v1 path, then
/// negotiates the pack with `command=fetch` POSTs (each round resends the
/// capability echo, all `want`s, and the accumulated `have`s) and demuxes the
/// side-band-64k `packfile` section. Lifted from the CLI's stateless v2 flow
/// (`http_ls_refs` / `http_negotiate_only_common` / `http_fetch_pack`), reusing
/// the v2 request framing factored out of [`crate::fetch`].
fn http_fetch_v2(
    client: &dyn HttpClient,
    local_git_dir: &Path,
    repo_url: &str,
    disc: &Discovery,
    opts: &FetchOptions,
    progress: &mut dyn Progress,
) -> Result<FetchOutcome> {
    let local_odb = open_odb(local_git_dir);
    // The v2 capability lines, as a `Vec<String>` for the `protocol_v2` /
    // `crate::fetch` helpers (each entry is one advertised capability line, e.g.
    // `agent=…`, `fetch=…`, `object-format=…`).
    let server_caps: Vec<String> = disc.caps.iter().cloned().collect();

    let post_url = upload_pack_url(repo_url);
    let content_type = format!("application/x-{UPLOAD_PACK}-request");
    let accept = format!("application/x-{UPLOAD_PACK}-result");
    // Pin v2 on every POST so the server runs its v2 serve loop for this request.
    let git_protocol = "version=2";

    // 1. Recover the ref map via `command=ls-refs`.
    let (remote_refs, head_symref) = {
        let req = crate::fetch::build_v2_ls_refs_request(
            &server_caps,
            &local_odb,
            opts.tags,
            &opts.refspecs,
        )?;
        let resp = client.post(&post_url, &content_type, &accept, &req, Some(git_protocol))?;
        let mut cur = Cursor::new(resp);
        crate::fetch::parse_v2_ls_refs_response(&mut cur)?
    };
    let default_branch = head_symref
        .as_deref()
        .map(|t| t.strip_prefix("refs/heads/").unwrap_or(t).to_owned());

    // 2. Parse refspecs.
    let mut positive: Vec<RefspecItem> = Vec::new();
    let mut negatives: Vec<RefspecItem> = Vec::new();
    for spec in &opts.refspecs {
        let item = parse_fetch_refspec(spec)
            .map_err(|e| Error::Message(format!("invalid refspec '{spec}': {e}")))?;
        if item.negative {
            negatives.push(item);
        } else {
            positive.push(item);
        }
    }
    for spec in &opts.negative_refspecs {
        let item = parse_fetch_refspec(spec)
            .map_err(|e| Error::Message(format!("invalid negative refspec '{spec}': {e}")))?;
        negatives.push(item);
    }

    // 3. Match refs to refspecs (shared with the v0/v1 path).
    let MatchPlan {
        mut matched,
        mut wants,
        mut seen,
    } = match_refspecs(&remote_refs, &positive, &negatives);

    // 4. TagMode: add tags (the wire `include-tag` capability brings tag objects
    // with the pack; All adds every advertised tag, Following adds them
    // provisionally and prunes unreachable ones after the pack lands).
    if opts.tags != TagMode::None {
        for (name, oid) in &remote_refs {
            if !name.starts_with("refs/tags/") {
                continue;
            }
            if seen.contains(name) || ref_excluded(name, &negatives) {
                continue;
            }
            seen.insert(name.clone());
            wants.insert(*oid);
            matched.push(crate::transfer::MatchedRef {
                remote_ref: name.clone(),
                local_ref: Some(name.clone()),
                oid: *oid,
                force: false,
                is_tag: true,
            });
        }
    }

    // 5. Wants → negotiate + ingest the pack. Normally the matched oids absent
    // locally; for a deepen/`--unshallow` request we must still `want` the tips
    // even if present so the server fills in ancestors past the old boundary.
    let local_shallow = crate::shallow::load_shallow_oids(local_git_dir)?;
    let shallow_request = opts.has_deepen_request() || !local_shallow.is_empty();
    let need: Vec<ObjectId> = if shallow_request {
        wants.iter().copied().collect()
    } else {
        wants
            .iter()
            .copied()
            .filter(|oid| !local_odb.exists(oid))
            .collect()
    };

    let mut shallow_update = crate::fetch::ShallowUpdate::default();

    if !need.is_empty() && !opts.dry_run {
        let deepen = crate::fetch::V2DeepenArgs::from_opts(opts, &local_shallow);
        let (pack, su) = negotiate_pack_v2_http(
            client,
            local_git_dir,
            &post_url,
            &content_type,
            &accept,
            git_protocol,
            &server_caps,
            &local_odb,
            &need,
            &deepen,
            progress,
        )?;
        shallow_update = su;
        if let Some(mut pack) = pack {
            pack.unpack_into(
                &local_odb,
                &crate::unpack_objects::UnpackOptions {
                    quiet: true,
                    ..Default::default()
                },
                progress,
                "did not receive a valid pack from v2 HTTP fetch",
            )?;
        }
    }

    // Apply shallow/unshallow boundary updates to the on-disk `shallow` file.
    if !opts.dry_run {
        crate::shallow::apply_shallow_updates(
            local_git_dir,
            &shallow_update.shallow,
            &shallow_update.unshallow,
        )?;
    }

    // 6. For TagMode::Following, drop tags whose target did not arrive.
    if opts.tags == TagMode::Following {
        retain_following_tags(&local_odb, &mut matched, &wants);
    }

    // 7. Classify + apply ref updates (shared with the v0/v1 path).
    let local_repo = if opts.dry_run {
        None
    } else {
        crate::repo::Repository::open(local_git_dir, None).ok()
    };

    let mut updates: Vec<RefUpdate> = Vec::new();
    if opts.prune {
        prune_tracking_refs(
            local_git_dir,
            &positive,
            &remote_refs,
            opts.dry_run,
            &mut updates,
        )?;
    }

    for m in &matched {
        let Some(local_ref) = &m.local_ref else {
            updates.push(RefUpdate {
                remote_ref: m.remote_ref.clone(),
                local_ref: None,
                old_oid: None,
                new_oid: Some(m.oid),
                mode: UpdateMode::NoChangeNeeded,
                note: Some("not stored (empty destination)".to_owned()),
            });
            continue;
        };
        let old = crate::refs::resolve_ref(local_git_dir, local_ref).ok();
        let mode = classify_update(old.as_ref(), &m.oid, m.force, m.is_tag, local_repo.as_ref());
        let write = matches!(
            mode,
            UpdateMode::New | UpdateMode::FastForward | UpdateMode::Forced
        );
        if write && !opts.dry_run {
            crate::refs::write_ref(local_git_dir, local_ref, &m.oid)?;
        }
        updates.push(RefUpdate {
            remote_ref: m.remote_ref.clone(),
            local_ref: Some(local_ref.clone()),
            old_oid: old,
            new_oid: Some(m.oid),
            mode,
            note: None,
        });
    }

    crate::net_trace::net_trace!("http_fetch (v2): done — {} ref update(s)", updates.len());
    Ok(FetchOutcome {
        updates,
        default_branch,
        new_shallow: shallow_update.shallow,
        new_unshallow: shallow_update.unshallow,
    })
}

/// Negotiate and download the pack for `wants` over stateless-RPC HTTP using
/// protocol v2 (`command=fetch`), returning the raw pack bytes.
///
/// Stateless: every POST resends the capability echo, every `want`, and all the
/// `have`s accumulated so far. The round structure mirrors the v0/v1 stateless
/// loop and the streaming v2 path:
///
/// * no local history → a single POST with `want`s + `done`, then read the
///   `packfile` section;
/// * otherwise → batched rounds that send `want`s + the growing have-prefix
///   *without* `done`, reading the `acknowledgments` section each time. When the
///   server replies `ready`, that same response carries the pack (read it and
///   stop). If the haves are exhausted without `ready`, a final POST sends every
///   have + `done` and reads the pack.
#[allow(clippy::too_many_arguments)]
fn negotiate_pack_v2_http(
    client: &dyn HttpClient,
    local_git_dir: &Path,
    post_url: &str,
    content_type: &str,
    accept: &str,
    git_protocol: &str,
    server_caps: &[String],
    local_odb: &crate::odb::Odb,
    wants: &[ObjectId],
    deepen: &crate::fetch::V2DeepenArgs,
    progress: &mut dyn Progress,
) -> Result<(PackStream, crate::fetch::ShallowUpdate)> {
    if wants.is_empty() {
        return Ok((None, crate::fetch::ShallowUpdate::default()));
    }
    let object_format = crate::fetch::v2_object_format(server_caps, local_odb);
    let cap_echo = protocol_v2::cap_lines_for_command_request(server_caps);
    let sideband_all = protocol_v2::fetch_supports_sideband_all(server_caps);

    // A deepen/shallow request does not offer haves (its objects bottom out at
    // grafts), forcing the single-round path so the server precedes the pack with
    // a `shallow-info` section.
    let shallow_request = deepen.is_shallow_request();

    // The ordered have list, built with the shared skipping-negotiator helper so
    // the wire offers match the streaming v2 path exactly. Empty for a shallow
    // request.
    let haves = if shallow_request {
        Vec::new()
    } else {
        crate::fetch::v2_local_haves(local_git_dir, wants)?
    };

    let mut pack = Vec::new();
    let mut shallow_update = crate::fetch::ShallowUpdate::default();

    // No local history: one POST, wants + done, then the pack.
    if haves.is_empty() {
        let mut req = Vec::new();
        crate::fetch::write_v2_fetch_request(
            &mut req,
            &object_format,
            &cap_echo,
            wants,
            &[],
            sideband_all,
            deepen,
            true,
        )?;
        let resp = client.post(post_url, content_type, accept, &req, Some(git_protocol))?;
        let mut cur = Cursor::new(resp);
        crate::fetch::read_v2_fetch_pack_response(
            &mut cur,
            &mut pack,
            &mut shallow_update,
            progress,
        )?;
        return Ok((pack_stream(pack, sideband_all), shallow_update));
    }

    // Batched negotiation: each round resends wants + the accumulated have prefix
    // (stateless) without `done`, reading the acknowledgments section. The flush
    // schedule matches `fetch-pack.c` (`next_flush`).
    const INITIAL_FLUSH: usize = 16;
    let mut flush_at: usize = INITIAL_FLUSH.min(haves.len());
    loop {
        if flush_at < haves.len() {
            // Non-final round: offer the have prefix [0..flush_at) without `done`.
            let mut req = Vec::new();
            crate::fetch::write_v2_fetch_request(
                &mut req,
                &object_format,
                &cap_echo,
                wants,
                &haves[..flush_at],
                sideband_all,
                deepen,
                false,
            )?;
            let resp = client.post(post_url, content_type, accept, &req, Some(git_protocol))?;
            let mut cur = Cursor::new(resp);
            let ack = crate::fetch::read_v2_acknowledgments(&mut cur)?;
            if let Some(round) = ack {
                if round.ready {
                    // The pack follows in this same response after the delimiter.
                    crate::fetch::read_v2_fetch_pack_response(
                        &mut cur,
                        &mut pack,
                        &mut shallow_update,
                        progress,
                    )?;
                    return Ok((pack_stream(pack, sideband_all), shallow_update));
                }
            } else {
                // Server skipped acknowledgments and went straight to the pack.
                crate::fetch::read_v2_fetch_pack_response(
                    &mut cur,
                    &mut pack,
                    &mut shallow_update,
                    progress,
                )?;
                return Ok((pack_stream(pack, sideband_all), shallow_update));
            }
            flush_at = next_flush(flush_at).min(haves.len());
            continue;
        }

        // Final round: send every have + `done`, then read the pack.
        let mut req = Vec::new();
        crate::fetch::write_v2_fetch_request(
            &mut req,
            &object_format,
            &cap_echo,
            wants,
            &haves,
            sideband_all,
            deepen,
            true,
        )?;
        let resp = client.post(post_url, content_type, accept, &req, Some(git_protocol))?;
        let mut cur = Cursor::new(resp);
        crate::fetch::read_v2_fetch_pack_response(
            &mut cur,
            &mut pack,
            &mut shallow_update,
            progress,
        )?;
        return Ok((pack_stream(pack, sideband_all), shallow_update));
    }
}

/// Drop provisional `Following` tags whose object did not arrive in the pack.
fn retain_following_tags(
    odb: &crate::odb::Odb,
    matched: &mut Vec<crate::transfer::MatchedRef>,
    wants: &HashSet<ObjectId>,
) {
    // No tag refs in the matched set → nothing to filter; skip the walk.
    if !matched.iter().any(|m| m.is_tag) {
        return;
    }
    let roots: Vec<ObjectId> = matched
        .iter()
        .filter(|m| !m.is_tag)
        .map(|m| m.oid)
        .collect();
    // Commit-level reachability suffices (and avoids walking every head's full
    // tree/blob closure — tens of seconds on a large repo). See the equivalent
    // path in `crate::fetch::retain_following_tags`.
    let closure = crate::fetch::reachable_commits(odb, &roots);
    matched.retain(|m| {
        if !m.is_tag {
            return true;
        }
        let peeled = peel_tag_target(odb, m.oid);
        let have = odb.exists(&m.oid);
        have && (closure.contains(&m.oid) || closure.contains(&peeled) || wants.contains(&peeled))
    });
}

fn peel_tag_target(odb: &crate::odb::Odb, oid: ObjectId) -> ObjectId {
    let mut current = oid;
    for _ in 0..16 {
        let Ok(obj) = odb.read(&current) else {
            return current;
        };
        if obj.kind != crate::objects::ObjectKind::Tag {
            return current;
        }
        match crate::objects::parse_tag(&obj.data) {
            Ok(t) => current = t.object,
            Err(_) => return current,
        }
    }
    current
}

/// Convenience: the unused-by-default [`Advertisement`] shape, exported so an
/// embedder can reuse the same structured view as the duplex transports.
pub fn discovery_advertisement(conn: &SmartHttpConnection) -> Advertisement {
    Advertisement {
        refs: conn.adv_refs.clone(),
        capabilities: conn.caps.clone(),
        head_symref: conn.head_symref.clone(),
        protocol_version: conn.protocol_version,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_redirect_strips_info_refs_and_query() {
        let base = "https://tangled.org/me/repo";
        // A redirect that preserves the query (real-world: tangled.org → knot host).
        assert_eq!(
            rebased_base_from_redirect(
                base,
                Some("https://knot.example/did:plc:xyz/info/refs?service=git-upload-pack")
            )
            .as_deref(),
            Some("https://knot.example/did:plc:xyz")
        );
        // A redirect that drops the query (still re-bases on the path suffix).
        assert_eq!(
            rebased_base_from_redirect(base, Some("https://host/smart/repo/info/refs")).as_deref(),
            Some("https://host/smart/repo")
        );
    }

    #[test]
    fn rebase_redirect_none_when_no_change_or_unknown() {
        let base = "https://host/smart/repo";
        // No final URL reported (client doesn't track redirects) → keep original base.
        assert_eq!(rebased_base_from_redirect(base, None), None);
        // Same base (no actual redirect) → None.
        assert_eq!(
            rebased_base_from_redirect(
                base,
                Some("https://host/smart/repo/info/refs?service=git-upload-pack")
            ),
            None
        );
        // A final URL that is not an `info/refs` request → None (don't guess).
        assert_eq!(
            rebased_base_from_redirect(base, Some("https://host/elsewhere")),
            None
        );
    }

    #[test]
    fn strips_smart_service_preamble() {
        let mut body = Vec::new();
        pkt_line::write_line_to_vec(&mut body, "# service=git-upload-pack\n").unwrap();
        body.extend_from_slice(b"0000");
        let oid = "1".repeat(40);
        let line = format!("{oid} refs/heads/main\0multi_ack_detailed side-band-64k");
        pkt_line::write_line_to_vec(&mut body, &line).unwrap();
        body.extend_from_slice(b"0000");

        let stripped = strip_service_advertisement(&body).unwrap();
        let disc = parse_advertisement(stripped).unwrap();
        assert_eq!(disc.protocol_version, 0);
        assert_eq!(disc.refs.len(), 1);
        assert_eq!(disc.refs[0].name, "refs/heads/main");
        assert!(disc.caps.contains("side-band-64k"));
    }

    #[test]
    fn parses_symref_and_caps() {
        let mut body = Vec::new();
        let main = "2".repeat(40);
        let head = format!(
            "{main} HEAD\0multi_ack_detailed symref=HEAD:refs/heads/main object-format=sha1"
        );
        pkt_line::write_line_to_vec(&mut body, &head).unwrap();
        let r = format!("{main} refs/heads/main");
        pkt_line::write_line_to_vec(&mut body, &r).unwrap();
        body.extend_from_slice(b"0000");

        let disc = parse_advertisement(&body).unwrap();
        assert_eq!(disc.head_symref.as_deref(), Some("refs/heads/main"));
        assert_eq!(disc.object_format, "sha1");
        // `parse_advertisement` keeps HEAD; the connection/fetch layer filters
        // HEAD and peeled `^{}` carriers. Both lines parse here.
        assert!(disc.refs.iter().any(|r| r.name == "HEAD"));
        assert!(disc.refs.iter().any(|r| r.name == "refs/heads/main"));
    }

    #[test]
    fn detects_v2_preamble() {
        let mut body = Vec::new();
        pkt_line::write_line_to_vec(&mut body, "version 2").unwrap();
        pkt_line::write_line_to_vec(&mut body, "ls-refs").unwrap();
        pkt_line::write_line_to_vec(&mut body, "object-format=sha256").unwrap();
        body.extend_from_slice(b"0000");
        let disc = parse_advertisement(&body).unwrap();
        assert_eq!(disc.protocol_version, 2);
        assert_eq!(disc.object_format, "sha256");
    }

    #[test]
    fn url_helpers() {
        assert_eq!(
            info_refs_url("http://h/r.git"),
            "http://h/r.git/info/refs?service=git-upload-pack"
        );
        assert_eq!(
            info_refs_url("http://h/r.git/"),
            "http://h/r.git/info/refs?service=git-upload-pack"
        );
        assert_eq!(
            upload_pack_url("http://h/r.git/"),
            "http://h/r.git/git-upload-pack"
        );
    }
}

#[cfg(test)]
mod streaming_reader_tests {
    use super::*;

    /// A packfile with no objects: a header and a trailing checksum. Enough to
    /// prove the framing is byte-exact.
    fn empty_pack() -> Vec<u8> {
        let mut pack = b"PACK".to_vec();
        pack.extend_from_slice(&2u32.to_be_bytes());
        pack.extend_from_slice(&0u32.to_be_bytes());
        pack.extend_from_slice(&[0u8; 20]);
        pack
    }

    /// Wrap a payload in a pkt-line, optionally with the side-band channel
    /// byte Git uses for a fetch response.
    fn pkt_line(payload: &[u8], channel: Option<u8>) -> Vec<u8> {
        let mut body = Vec::new();
        if let Some(channel) = channel {
            body.push(channel);
        }
        body.extend_from_slice(payload);
        let mut out = format!("{:04x}", body.len() + 4).into_bytes();
        out.extend_from_slice(&body);
        out
    }

    /// A body that hands out one byte at a time, the worst case for a reader
    /// that is counting on chunk boundaries meaning nothing.
    struct Dribble {
        bytes: Vec<u8>,
        at: usize,
    }

    impl HttpBody for Dribble {
        fn next_chunk(&mut self) -> Result<Option<Vec<u8>>> {
            if self.at >= self.bytes.len() {
                return Ok(None);
            }
            let end = (self.at + 1).min(self.bytes.len());
            let chunk = self.bytes[self.at..end].to_vec();
            self.at = end;
            Ok(Some(chunk))
        }
    }

    fn reader(body: Vec<u8>, sideband: bool) -> SidebandPackReader {
        let dribble = Dribble { bytes: body, at: 0 };
        SidebandPackReader::new(Box::new(BodyReader::new(Box::new(dribble))), sideband)
    }

    /// The control section, then the pack, then the pack is read back and must
    /// come out byte for byte.
    #[test]
    fn pack_after_control_section_reads_back_exactly() {
        let pack = empty_pack();
        let mut body = pkt_line(b"NAK\n", None);
        body.extend_from_slice(&pkt_line(&pack, Some(1)));
        let mut src = reader(body, true);

        // Drive the control parser the way read_stateless_response does.
        src.mark();
        let first = src.read_control_payload().unwrap().unwrap();
        assert_eq!(first, b"NAK\n");
        src.rewind();
        src.begin_pack();

        let mut got = Vec::new();
        let mut buffer = [0u8; 7];
        loop {
            let n = std::io::Read::read(&mut src, &mut buffer).unwrap();
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buffer[..n]);
        }
        assert_eq!(got, pack, "the pack must come back byte for byte");
    }

    /// The point of the whole type: a large pack goes through without ever
    /// being held whole.
    ///
    /// A 256 MiB pack is built, wrapped in side-band packets, and read back
    /// 8 KiB at a time. The bytes must come back identical, and what the
    /// reader retains must stay near the read size no matter how big the
    /// pack gets -- that is the difference between streaming and buffering,
    /// and it is asserted here because the failure mode is invisible in a
    /// small test.
    #[test]
    fn a_large_pack_streams_without_being_held_whole() {
        const PACK_BYTES: usize = 256 * 1024 * 1024;
        let mut pack = b"PACK".to_vec();
        pack.extend_from_slice(&2u32.to_be_bytes());
        pack.extend_from_slice(&0u32.to_be_bytes());
        let filler = vec![b'a'; PACK_BYTES - pack.len() - 20];
        pack.extend_from_slice(&filler);
        pack.extend_from_slice(&[0u8; 20]);
        let expected = pack.len();

        // 64 KiB side-band packets, the size GitHub uses.
        let mut body = pkt_line(b"NAK\n", None);
        for piece in pack.chunks(60 * 1024) {
            body.extend_from_slice(&pkt_line(piece, Some(1)));
        }
        drop(pack);

        let dribble = Dribble { bytes: body, at: 0 };
        let mut src = SidebandPackReader::new(
            Box::new(BodyReader::new(Box::new(dribble))),
            true,
        );
        src.mark();
        let _ = src.read_control_payload().unwrap().unwrap();
        src.rewind();
        src.begin_pack();

        let mut total = 0usize;
        let mut buffer = [0u8; 8 * 1024];
        let mut peak = 0usize;
        loop {
            let n = std::io::Read::read(&mut src, &mut buffer).unwrap();
            if n == 0 {
                break;
            }
            total += n;
            peak = peak.max(src.retained_bytes());
        }
        assert_eq!(total, expected, "every byte of the pack must come back");
        assert!(
            peak < 1024 * 1024,
            "the reader retained {peak} bytes at peak, which is not streaming",
        );
        assert!(
            peak < expected / 64,
            "retaining {peak} of {expected} bytes is buffering, not streaming",
        );
    }
    /// The same thing with the whole pack delivered in side-band chunks, which
    /// is how a real server splits a large packfile.
    #[test]
    fn pack_split_across_many_sideband_chunks() {
        let pack = empty_pack();
        let mut body = pkt_line(b"NAK\n", None);
        for piece in pack.chunks(5) {
            body.extend_from_slice(&pkt_line(piece, Some(1)));
        }
        let mut src = reader(body, true);

        src.mark();
        let _ = src.read_control_payload().unwrap().unwrap();
        src.rewind();
        src.begin_pack();

        let mut got = Vec::new();
        let mut buffer = [0u8; 3];
        loop {
            let n = std::io::Read::read(&mut src, &mut buffer).unwrap();
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buffer[..n]);
        }
        assert_eq!(got, pack, "chunking must not change the bytes");
    }
}
