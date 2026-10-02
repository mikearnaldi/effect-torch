//! Host-only structural witness for the combined99 sampler boundary.
use effect_torch_compiler::*;
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{DType, StorageMetadata};
struct Target {
    device: Device,
    fingerprint: TargetFingerprint,
}
impl TargetDTypeCapabilities for Target {
    fn device(&self) -> &Device {
        &self.device
    }
    fn fingerprint(&self) -> &TargetFingerprint {
        &self.fingerprint
    }
    fn policy_revision(&self) -> u64 {
        1
    }
    fn storage_support(&self, _: &ValueSpec<'_>) -> StorageSupport {
        StorageSupport::Supported
    }
    fn classify_node(&self, s: &OperationDTypeSpec<'_>) -> DTypeDisposition {
        DTypeDisposition::Native(s.native_execution())
    }
    fn classify_region(&self, s: &RegionDTypeSpec<'_>) -> DTypeDisposition {
        if matches!(s.region, NativeRegion::Elementwise(_)) {
            DTypeDisposition::Native(s.native_execution())
        } else {
            DTypeDisposition::Unsupported(UnsupportedDType::new(
                DTypeRequirement::Region,
                "witness admits only elementwise regions",
            ))
        }
    }
}
fn main() {
    let target = Target {
        device: Device::Cuda(0),
        fingerprint: TargetFingerprint::new(TargetBackend::Cuda, "sampler99-structural", 1),
    };
    let input = |slot, shape| {
        Node::new(NodeKind::Input {
            slot,
            shape,
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    };
    let constant = |value| {
        Node::new(NodeKind::Full {
            shape: vec![],
            value,
            dtype: DType::F32,
            device: Device::Cuda(0),
        })
        .unwrap()
    };
    let scaled = Node::new(NodeKind::Mul {
        a: input(0, vec![1, 256, 262144]),
        b: constant(1.0 / 30.0),
    })
    .unwrap();
    let capped = Node::new(NodeKind::Tanh { a: scaled }).unwrap();
    let logits = Node::new(NodeKind::Mul {
        a: capped,
        b: constant(30.0),
    })
    .unwrap();
    let processed = Node::new(NodeKind::Div {
        a: logits.clone(),
        b: input(1, vec![]),
    })
    .unwrap();
    let feedback = Node::new(NodeKind::Cast {
        a: processed.clone(),
        dtype: DType::BF16,
    })
    .unwrap();
    let arg = Node::new(NodeKind::Argmax {
        a: processed.clone(),
        dim: 2,
    })
    .unwrap();
    for rooted in [false, true] {
        let mut roots = vec![feedback.clone(), arg.clone()];
        if rooted {
            roots.push(logits.clone());
        }
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let driver = CompilerDriver::new(&prepared, &target).unwrap();
        let division = prepared
            .index
            .order
            .iter()
            .position(|n| n.id == processed.id)
            .unwrap();
        let absorbed = driver.optimization().node_region[division].is_some();
        assert_eq!(absorbed, !rooted);
        println!("{{\"logitsRooted\":{rooted},\"divisionAbsorbedByElementwise\":{absorbed}}}");
    }
}
