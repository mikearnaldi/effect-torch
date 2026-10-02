//! Bounded materialization of existing immutable literal nodes, without compilation.
use super::*;
use effect_torch_runtime::CancellationFlag;

fn validate(roots: &[Arc<Node>], ordinal: u32) -> std::result::Result<(), String> {
    if roots.is_empty() || roots.len() > 8 {
        return Err("literal89: expected one to eight roots".into());
    }
    let mut bytes = 0usize;
    for node in roots {
        if node.device != Device::Cuda(ordinal) {
            return Err("literal89: graph placement differs from runtime".into());
        }
        match &node.kind {
            NodeKind::Full {
                shape,
                dtype: DType::F32,
                value,
                ..
            } if shape.is_empty() && value.is_finite() => {}
            NodeKind::FromBytes {
                dtype: DType::U32, ..
            } => {}
            _ => return Err("literal89: unsupported literal root".into()),
        }
        bytes = bytes
            .checked_add(node.value_spec().canonical_geometry()?.byte_len)
            .ok_or("literal89: byte count overflow")?;
        if bytes > 64 * 1024 {
            return Err("literal89: literals exceed 64 KiB".into());
        }
    }
    Ok(())
}

pub(super) fn materialize(
    device: Arc<CudaDevice>,
    roots: &[Arc<Node>],
    cancelled: &CancellationFlag,
) -> std::result::Result<Vec<CudaValue>, String> {
    materialize_inner(
        device,
        roots,
        cancelled,
        #[cfg(test)]
        None,
    )
}

fn materialize_inner(
    device: Arc<CudaDevice>,
    roots: &[Arc<Node>],
    cancelled: &CancellationFlag,
    #[cfg(test)] after_upload: Option<&dyn Fn()>,
) -> std::result::Result<Vec<CudaValue>, String> {
    validate(roots, device.ordinal)?;
    if cancelled.is_cancelled() {
        return Err("operation aborted".into());
    }
    // Serialize with all stream-capture paths through completion and cleanup.
    let _gate = device
        .graph_execution
        .lock()
        .map_err(|_| "literal89: CUDA graph execution lock poisoned")?;
    if cancelled.is_cancelled() {
        return Err("operation aborted".into());
    }
    // Keep all owners until the upload stream drains, including partial failure.
    let mut values: Vec<CudaValue> = Vec::with_capacity(roots.len());
    let result = (|| {
        for (index, node) in roots.iter().enumerate() {
            if cancelled.is_cancelled() {
                return Err("operation aborted".to_string());
            }
            let value = if let Some(previous) =
                roots[..index].iter().position(|p| Arc::ptr_eq(p, node))
            {
                values[previous].clone()
            } else {
                match &node.kind {
                    NodeKind::Full { value, .. } => {
                        CudaValue::from_host(device.clone(), Vec::new(), DType::F32, &[*value])?
                    }
                    NodeKind::FromBytes {
                        data, shape, dtype, ..
                    } => CudaValue::from_dense_bytes(device.clone(), shape.clone(), *dtype, data)?,
                    _ => unreachable!("validated literal"),
                }
            };
            values.push(value);
            #[cfg(test)]
            if let Some(hook) = after_upload {
                hook();
            }
        }
        Ok(())
    })();
    if let Err(error) = device.stream.synchronize() {
        // Completion is unknown: never release potentially in-flight allocations.
        std::mem::forget(values);
        return Err(error.to_string());
    }
    result?;
    if cancelled.is_cancelled() {
        return Err("operation aborted".into());
    }
    Ok(values)
}

pub(super) async fn execute(
    device: Arc<CudaDevice>,
    roots: Vec<Arc<Node>>,
    token: Option<&CancellationToken>,
) -> Result<Vec<NativeTensor>> {
    validate(&roots, device.ordinal).map_err(invalid)?;
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let notify = token.map(|token| token.notify.clone());
    run_compute(state, notify, move |cancelled, _| {
        materialize(device, &roots, cancelled)
            .map(|values| values.into_iter().map(NativeTensor::wrap).collect())
            .map_err(|message| {
                if cancelled.is_cancelled() {
                    Error::new(Status::Cancelled, "operation aborted")
                } else {
                    failure(message)
                }
            })
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    fn full(dtype: DType, shape: Vec<usize>, value: f64) -> Arc<Node> {
        Node::new(NodeKind::Full {
            shape,
            dtype,
            value,
            device: Device::Cuda(0),
        })
        .unwrap()
    }
    #[test]
    fn literal89_admission_is_bounded_and_placement_exact() {
        let root = full(DType::F32, vec![], 0.7);
        assert!(validate(std::slice::from_ref(&root), 0).is_ok());
        assert!(validate(std::slice::from_ref(&root), 1).is_err());
        assert!(validate(&[], 0).is_err());
        assert!(validate(&vec![root; 9], 0).is_err());
        for root in [
            full(DType::F64, vec![], 1.),
            full(DType::F32, vec![1], 1.),
            full(DType::F32, vec![], f64::NAN),
        ] {
            assert!(validate(&[root], 0).is_err());
        }
        for count in [16384, 16385] {
            let root = Node::new(NodeKind::FromBytes {
                data: vec![0; count * 4],
                shape: vec![count],
                dtype: DType::U32,
                device: Device::Cuda(0),
            })
            .unwrap();
            assert_eq!(validate(&[root], 0).is_ok(), count == 16384);
        }
    }
    #[test]
    #[ignore = "requires CUDA; literal89 snapshot, retained outputs and cancellation"]
    fn literal89_hardware_snapshot_retention_and_precancel() {
        let device = CudaDevice::get(0).unwrap();
        let mut bytes = vec![1u32, 19, u32::MAX]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let root = Node::new(NodeKind::FromBytes {
            data: bytes.clone(),
            shape: vec![3],
            dtype: DType::U32,
            device: Device::Cuda(0),
        })
        .unwrap();
        let expected = bytes.clone();
        bytes.fill(0);
        let roots = vec![root.clone(), full(DType::F32, vec![], -0.), root];
        let cancel = CancellationFlag::new();
        cancel.cancel();
        assert!(materialize(device.clone(), &roots, &cancel).is_err());
        let during = CancellationFlag::new();
        assert!(
            materialize_inner(device.clone(), &roots, &during, Some(&|| during.cancel())).is_err()
        );
        let mut first = materialize(device.clone(), &roots, &CancellationFlag::new()).unwrap();
        let second = materialize(device, &roots, &CancellationFlag::new()).unwrap();
        drop(roots);
        drop(second);
        first.remove(0);
        assert_eq!(
            first[0].read_storage_bytes().unwrap(),
            (-0f32).to_le_bytes()
        );
        assert_eq!(first[1].read_storage_bytes().unwrap(), expected);
    }
}
