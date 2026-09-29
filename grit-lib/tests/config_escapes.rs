//! Regression tests for backslash-escape handling in git config values.
//!
//! A value may begin with an escaped quote. That is not exotic: git's own
//! opendiff helper is configured as
//!
//!     [diff]
//!         cmd = opendiff \&quot;$LOCAL\&quot; \&quot;$REMOTE\&quot;
//!
//! and users copy that line into their ~/.gitconfig. Treating the leading
//! escaped quote as an opening quote made the parser believe the value never
//! closed, which rejected the whole file - so a plain status call failed with
//! 'bad config line 9' for anyone who had it.

use grit_lib::config::{ConfigFile, ConfigScope};
use std::io::Write;

fn parse(contents: &str) -> Result<Vec<(String, String)>, String> {
    let dir = std::env::temp_dir().join(format!(
        "grit-cfg-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("config");
    let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    file.write_all(contents.as_bytes()).map_err(|e| e.to_string())?;
    drop(file);

    let loaded = ConfigFile::from_path(&path, ConfigScope::Global).map_err(|e| format!("{e:?}"))?;
    let _ = std::fs::remove_dir_all(&dir);
    let Some(loaded) = loaded else {
        return Err("config file did not load".to_string());
    };
    Ok(loaded
        .entries
        .iter()
        .map(|e| (e.key.clone(), e.value.clone().unwrap_or_default()))
        .collect())
}

#[test]
fn value_may_start_with_an_escaped_quote() {
    // The exact line from git's own opendiff documentation.
    let text = "[diff]\n    cmd = opendiff \\\u{22}$LOCAL\\\u{22} \\\u{22}$REMOTE\\\u{22}\n";
    let entries = parse(text).expect("a value starting with an escaped quote should parse");
    assert_eq!(entries.len(), 1, "expected one entry, got {entries:?}");
    assert_eq!(entries[0].0, "diff.cmd");
    assert!(entries[0].1.contains("$LOCAL"), "got {:?}", entries[0].1);
    assert!(entries[0].1.contains("$REMOTE"), "got {:?}", entries[0].1);
}

#[test]
fn trailing_comment_after_escaped_quotes_is_still_a_comment() {
    let text = "[diff]\n    cmd = opendiff \\\u{22}$LOCAL\\\u{22} \\\u{22}$REMOTE\\\u{22} # keep the diff tool\n";
    let entries = parse(text).expect("a comment after escaped quotes should parse");
    assert!(
        !entries[0].1.contains("keep the diff tool"),
        "the trailing comment leaked into the value: {:?}",
        entries[0].1
    );
}

#[test]
fn genuinely_unterminated_quote_is_still_rejected() {
    // The fix must not disable the balance check: a value that really never
    // closes is still an error. This one uses a BARE quote, not an escaped one.
    let text = "[diff]\n    cmd = opendiff \"$LOCAL\n";
    assert!(
        parse(text).is_err(),
        "an unterminated quote must still be an error"
    );
}
