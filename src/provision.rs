use std::sync::{Arc, Mutex};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

use crate::{
    Inner, PosError, PosState, Result,
    state::{self, VersionFile, SCHEMA_VERSION},
    perceptron::PerceptronTagger,
};

pub async fn run(
    inner: Arc<Mutex<Inner>>,
    on_progress: impl Fn(PosState) + Send + 'static,
) -> Result<()> {
    {
        let guard = inner.lock().unwrap();
        if state::is_installed_for(&guard.config) {
            info!("[pos] already installed for {}", guard.config.lang);
            drop(guard);
            return try_load_model(inner);
        }
        match &guard.state {
            PosState::Downloading { .. } | PosState::Indexing => {
                info!("[pos] provision already in flight for {}", guard.config.lang);
                return Ok(());
            }
            _ => {}
        }
        std::fs::create_dir_all(&guard.config.data_dir)?;
    }

    let (url, sha256_expected, lang) = {
        let g = inner.lock().unwrap();
        (
            g.config.source_url.clone(),
            g.config.source_sha256.clone(),
            g.config.lang.clone(),
        )
    };

    let part_path = state::part_path(&inner.lock().unwrap().config);

    info!("[pos] downloading {} from {}", lang, url);
    let client = reqwest::Client::new();
    let resp = client.get(&url).send().await.map_err(PosError::Http)?;
    let total = resp.content_length();

    set_state(&inner, PosState::Downloading { downloaded: 0, total });
    on_progress(PosState::Downloading { downloaded: 0, total });

    let mut file = tokio::fs::File::create(&part_path).await?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut buf: Vec<u8> = Vec::new();

    use futures_util::StreamExt;
    let mut byte_stream = resp.bytes_stream();

    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.map_err(PosError::Http)?;
        hasher.update(&chunk);
        downloaded += chunk.len() as u64;
        buf.extend_from_slice(&chunk);
        file.write_all(&chunk).await?;
        let s = PosState::Downloading { downloaded, total };
        set_state(&inner, s.clone());
        on_progress(s);
    }
    file.flush().await?;
    drop(file);

    let actual = format!("{:x}", hasher.finalize());
    if actual != sha256_expected {
        let _ = std::fs::remove_file(&part_path);
        warn!("[pos] checksum mismatch for {}: expected {} got {}", lang, sha256_expected, actual);
        let err = PosError::ChecksumMismatch {
            expected: sha256_expected,
            actual,
        };
        set_state(&inner, PosState::Error { message: err.to_string() });
        return Err(err);
    }
    info!("[pos] checksum ok for {}", lang);

    set_state(&inner, PosState::Indexing);
    on_progress(PosState::Indexing);

    let dest = inner.lock().unwrap().config.data_dir.clone();
    unpack_tar_gz(&buf, &dest).map_err(|e| PosError::Model(e.to_string()))?;

    let ver_path = state::version_path(&inner.lock().unwrap().config);
    let version = VersionFile {
        lang: lang.clone(),
        source_sha256: sha256_expected,
        schema_version: SCHEMA_VERSION,
    };
    std::fs::write(&ver_path, serde_json::to_string_pretty(&version).unwrap())?;

    let _ = std::fs::remove_file(&part_path);

    try_load_model(inner.clone())?;
    on_progress(PosState::Ready);
    info!("[pos] provision complete for {}", lang);
    Ok(())
}

fn unpack_tar_gz(data: &[u8], dest_dir: &std::path::Path) -> std::io::Result<()> {
    let cursor = std::io::Cursor::new(data);
    let gz = flate2::read::GzDecoder::new(cursor);
    let mut archive = tar::Archive::new(gz);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let filename = path.file_name().unwrap_or_default().to_os_string();
        let dest = dest_dir.join(filename);
        let mut out = std::fs::File::create(&dest)?;
        std::io::copy(&mut entry, &mut out)?;
    }
    Ok(())
}

pub fn try_load_model(inner: Arc<Mutex<Inner>>) -> Result<()> {
    let (weights_path, lang) = {
        let g = inner.lock().unwrap();
        (state::weights_path(&g.config), g.config.lang.clone())
    };

    match PerceptronTagger::load(&weights_path, &lang) {
        Ok(tagger) => {
            let mut g = inner.lock().unwrap();
            g.tagger = Some(tagger);
            g.state = PosState::Ready;
            info!("[pos] model loaded for {}", lang);
            Ok(())
        }
        Err(e) => {
            let mut g = inner.lock().unwrap();
            g.state = PosState::Error { message: e.to_string() };
            Err(e)
        }
    }
}

fn set_state(inner: &Arc<Mutex<Inner>>, state: PosState) {
    inner.lock().unwrap().state = state;
}
