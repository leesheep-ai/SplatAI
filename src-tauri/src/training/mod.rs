use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductMode {
    Compatible,
    Fast,
    Quality,
    Ultimate,
    Geometry,
    LargeScene,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Brush,
    Gsplat,
    FastGs,
    ThreeDgrut,
    FastPgsr,
    OctreeGs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub dataset: PathBuf,
    pub output_dir: PathBuf,
    pub iterations: u32,
    pub max_resolution: Option<u32>,
    pub device: Option<String>,
    pub enable_3dgut: bool,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessSpec {
    pub executable: PathBuf,
    pub args: Vec<String>,
    pub working_directory: Option<PathBuf>,
    pub environment: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeHealth {
    pub python_ok: bool,
    pub torch_ok: bool,
    pub gsplat_ok: bool,
    pub cuda_available: bool,
    pub torch_version: Option<String>,
    pub cuda_version: Option<String>,
    pub gsplat_version: Option<String>,
    pub gpu_name: Option<String>,
    pub vram_bytes: Option<u64>,
    pub compute_capability: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TrainingError {
    #[error("product mode is not implemented yet: {0:?}")]
    ModeUnavailable(ProductMode),
    #[error("iterations must be positive")]
    InvalidIterations,
}

pub fn select_backend(mode: ProductMode) -> Result<BackendKind, TrainingError> {
    match mode {
        ProductMode::Compatible => Ok(BackendKind::Brush),
        ProductMode::Quality | ProductMode::Ultimate => Ok(BackendKind::Gsplat),
        other => Err(TrainingError::ModeUnavailable(other)),
    }
}

pub fn recommend_mode(health: &RuntimeHealth) -> ProductMode {
    const GIB: u64 = 1024 * 1024 * 1024;

    if !(health.python_ok && health.torch_ok && health.gsplat_ok && health.cuda_available) {
        return ProductMode::Compatible;
    }

    match health.vram_bytes.unwrap_or(0) {
        vram if vram >= 16 * GIB => ProductMode::Ultimate,
        vram if vram >= 8 * GIB => ProductMode::Quality,
        _ => ProductMode::Compatible,
    }
}

pub fn build_brush_command(
    executable: impl Into<PathBuf>,
    request: &TrainingRequest,
) -> Result<ProcessSpec, TrainingError> {
    if request.iterations == 0 {
        return Err(TrainingError::InvalidIterations);
    }

    let mut args = vec![
        "--total-steps".into(),
        request.iterations.to_string(),
    ];

    if let Some(max_resolution) = request.max_resolution {
        args.extend(["--max-resolution".into(), max_resolution.to_string()]);
    }

    args.extend([
        "--export-every".into(),
        request.iterations.to_string(),
        "--export-path".into(),
        request.output_dir.display().to_string(),
        "--export-name".into(),
        "final.ply.tmp".into(),
    ]);
    args.extend(request.extra_args.clone());
    args.push(request.dataset.display().to_string());

    Ok(ProcessSpec {
        executable: executable.into(),
        args,
        working_directory: Some(request.output_dir.clone()),
        environment: BTreeMap::new(),
    })
}

pub fn build_gsplat_command(
    python_executable: impl Into<PathBuf>,
    trainer_script: impl Into<PathBuf>,
    request: &TrainingRequest,
) -> Result<ProcessSpec, TrainingError> {
    if request.iterations == 0 {
        return Err(TrainingError::InvalidIterations);
    }

    let trainer_script = trainer_script.into();
    let mut args = vec![
        trainer_script.display().to_string(),
        "mcmc".into(),
        "--data-dir".into(),
        request.dataset.display().to_string(),
        "--result-dir".into(),
        request.output_dir.display().to_string(),
        "--max-steps".into(),
        request.iterations.to_string(),
        "--save-ply".into(),
        "True".into(),
        "--ply-steps".into(),
        request.iterations.to_string(),
        "--disable-viewer".into(),
        "True".into(),
        "--disable-video".into(),
        "True".into(),
    ];

    if request.enable_3dgut {
        args.extend([
            "--with-ut".into(),
            "True".into(),
            "--with-eval3d".into(),
            "True".into(),
        ]);
    }

    args.extend(request.extra_args.clone());

    let mut environment = BTreeMap::new();
    if let Some(device) = &request.device {
        if let Some(index) = device.strip_prefix("cuda:") {
            environment.insert("CUDA_VISIBLE_DEVICES".into(), index.into());
        }
    }

    Ok(ProcessSpec {
        executable: python_executable.into(),
        args,
        working_directory: None,
        environment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> TrainingRequest {
        TrainingRequest {
            dataset: PathBuf::from("dataset"),
            output_dir: PathBuf::from("output"),
            iterations: 30_000,
            max_resolution: Some(1600),
            device: Some("cuda:1".into()),
            enable_3dgut: true,
            extra_args: Vec::new(),
        }
    }

    #[test]
    fn brush_contract_matches_existing_cli_shape() {
        let spec = build_brush_command("brush_app", &request()).unwrap();
        assert_eq!(spec.executable, PathBuf::from("brush_app"));
        assert!(spec.args.windows(2).any(|w| w == ["--total-steps", "30000"]));
        assert!(spec.args.windows(2).any(|w| w == ["--max-resolution", "1600"]));
        assert!(spec.args.windows(2).any(|w| w == ["--export-name", "final.ply.tmp"]));
        assert_eq!(spec.args.last().unwrap(), "dataset");
    }

    #[test]
    fn gsplat_uses_mcmc_and_full_3dgut_flags() {
        let spec = build_gsplat_command(
            "python",
            "engines/gsplat/simple_trainer.py",
            &request(),
        )
        .unwrap();
        assert_eq!(spec.args[1], "mcmc");
        assert!(spec.args.contains(&"--with-ut".into()));
        assert!(spec.args.contains(&"--with-eval3d".into()));
        assert_eq!(spec.environment.get("CUDA_VISIBLE_DEVICES"), Some(&"1".into()));
    }

    #[test]
    fn recommendation_is_conservative() {
        let health = RuntimeHealth {
            python_ok: true,
            torch_ok: true,
            gsplat_ok: true,
            cuda_available: true,
            torch_version: None,
            cuda_version: None,
            gsplat_version: None,
            gpu_name: Some("GPU".into()),
            vram_bytes: Some(16 * 1024 * 1024 * 1024),
            compute_capability: None,
            error: None,
        };
        assert_eq!(recommend_mode(&health), ProductMode::Ultimate);
    }
}
