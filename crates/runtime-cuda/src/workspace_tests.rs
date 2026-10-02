use super::*;
use crate::buffer::validate_segment_range;
use effect_torch_runtime::{CancellationFlag, DType, SegmentId, StorageMetadata};

#[test]
fn planned_segment_geometry_rejects_overflow_alignment_and_out_of_bounds() {
    assert_eq!(validate_segment_range::<u32>(8, 3, 20).unwrap(), 20);
    assert_eq!(validate_segment_range::<u32>(20, 0, 20).unwrap(), 20);
    assert!(validate_segment_range::<u32>(1, 0, 0)
        .unwrap_err()
        .contains("not aligned"));
    assert!(validate_segment_range::<u64>(0, usize::MAX, usize::MAX)
        .unwrap_err()
        .contains("byte size overflowed"));
    assert!(validate_segment_range::<u8>(usize::MAX, 1, usize::MAX)
        .unwrap_err()
        .contains("range overflowed"));
    assert!(validate_segment_range::<u16>(20, 1, 20)
        .unwrap_err()
        .contains("exceeds segment size"));
}

fn segments() -> Vec<SegmentDecl<CudaMemorySpace>> {
    [
        SegmentOwnership::Workspace,
        SegmentOwnership::ProvisionalOutput,
        SegmentOwnership::StateTransaction,
    ]
    .into_iter()
    .map(|ownership| SegmentDecl {
        bytes: 256,
        alignment: 256,
        memory_space: CudaMemorySpace::Device,
        ownership,
    })
    .collect()
}
fn location(segment: usize, offset: usize, bytes: usize) -> Location {
    Location::Segment {
        segment: SegmentId::from_index(segment).unwrap(),
        offset,
        bytes,
    }
}

#[test]
#[ignore = "requires CUDA"]
fn cached_segment_views_match_original_bounds_aliases_and_retained_leases() {
    let device = CudaDevice::get(0).unwrap();
    for cached in [false, true] {
        for segment_index in [1, 2] {
            let frame = acquire_with_policy(0, &segments(), cached).unwrap();
            assert!(frame
                .segments
                .iter()
                .all(|s| s.full_buffer.is_some() == cached));
            // Normalized alias: declared root begins16bytes in, alias adds8bytes.
            let loc = location(segment_index, 16, 128);
            // Test-only source counter, absent from benchmark builds: cached views
            // perform no repeated CudaSlice::device_ptr access or read-event guard.
            let reads_before = crate::buffer::planned_address_reads();
            for _ in 0..100 {
                let derived = frame.buffer::<u32>(&loc, 8, 8).unwrap();
                assert_eq!(derived.len(), 8);
            }
            assert_eq!(
                crate::buffer::planned_address_reads() - reads_before,
                if cached { 0 } else { 100 }
            );

            let mut view = frame.buffer::<u32>(&loc, 8, 8).unwrap();
            let reference = CudaBuffer::<u32>::from_segment(
                frame.segments[segment_index].owner.clone(),
                24,
                8,
                frame.segments[segment_index].retention.clone(),
            )
            .unwrap();
            assert_eq!(view.address(), reference.address());
            assert_eq!(view.allocation(), reference.allocation());
            device
                .stream
                .memcpy_htod(&[3u32, 5, 7, 11, 13, 17, 19, 23], &mut view)
                .unwrap();
            device.stream.synchronize().unwrap();
            assert_eq!(
                device.stream.clone_dtoh(&reference).unwrap(),
                vec![3, 5, 7, 11, 13, 17, 19, 23]
            );
            let lease = Arc::downgrade(frame.segments[segment_index].retention.as_ref().unwrap());
            drop(reference);
            drop(frame);
            assert!(lease.upgrade().is_some());
            let next = acquire_with_policy(0, &segments(), cached).unwrap();
            assert_ne!(
                next.buffer::<u32>(&loc, 8, 8).unwrap().allocation().0,
                view.allocation().0
            );
            assert_eq!(
                device.stream.clone_dtoh(&view).unwrap(),
                vec![3, 5, 7, 11, 13, 17, 19, 23]
            );
            drop(next);
            drop(view);
            assert!(lease.upgrade().is_none());
        }
    }
    let off = acquire_with_policy(0, &segments(), false).unwrap();
    let on = acquire_with_policy(0, &segments(), true).unwrap();
    for (loc, extra, len) in [
        (location(0, 0, 256), 1, 1),
        (location(0, 0, 8), 8, 1),
        (location(0, usize::MAX, 8), 1, 1),
        (location(99, 0, 8), 0, 1),
        (location(0, 252, 8), 0, 2),
        (Location::External { slot: 0 }, 0, 1),
        (location(0, 0, 256), 0, usize::MAX),
    ] {
        assert_eq!(
            off.buffer::<u32>(&loc, extra, len).err().unwrap(),
            on.buffer::<u32>(&loc, extra, len).err().unwrap()
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn cached_segment_concurrent_frames_have_independent_backing() {
    let barrier = Arc::new(std::sync::Barrier::new(5));
    let threads = (0..4)
        .map(|_| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let frame = acquire_with_policy(0, &segments(), true).unwrap();
                let identities = frame
                    .segments
                    .iter()
                    .map(|s| s.full_buffer.as_ref().unwrap().allocation().0)
                    .collect::<Vec<_>>();
                barrier.wait();
                barrier.wait();
                drop(frame);
                identities
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    barrier.wait();
    let all = threads
        .into_iter()
        .flat_map(|t| t.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        all.iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        all.len()
    );
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_CACHE_SEGMENT_BUFFERS=1"]
fn cached_segment_execution_cancellation_error_recovery_and_retained_output() {
    use effect_torch_graph::{Device, Node, NodeKind};
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_CACHE_SEGMENT_BUFFERS").as_deref(),
        Ok("1")
    );
    let input = |slot| {
        Node::new(NodeKind::Input {
            slot,
            shape: vec![8],
            dtype: DType::I64,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    };
    let a = input(0);
    let doubled = Node::new(NodeKind::Add { a: a.clone(), b: a }).unwrap();
    let output = Node::new(NodeKind::Div {
        a: doubled,
        b: input(1),
    })
    .unwrap();
    let executable = crate::compile(vec![output], 0).unwrap();
    let device = CudaDevice::get(0).unwrap();
    let value =
        |v| crate::CudaValue::from_host(device.clone(), vec![8], DType::I64, &vec![v; 8]).unwrap();
    let inputs = [value(6.), value(2.)];
    let cancelled = CancellationFlag::new();
    cancelled.cancel();
    assert!(executable.execute(&inputs, &[], &cancelled).is_err());
    let retained = executable
        .execute(&inputs, &[], &CancellationFlag::new())
        .unwrap()
        .remove(0);
    let bytes = retained.read_storage_bytes().unwrap();
    assert!(executable
        .execute(&[value(6.), value(0.)], &[], &CancellationFlag::new())
        .is_err());
    let recovered = executable
        .execute(&inputs, &[], &CancellationFlag::new())
        .unwrap()
        .remove(0);
    assert_eq!(recovered.read_storage_bytes().unwrap(), bytes);
    assert_eq!(retained.read_storage_bytes().unwrap(), bytes);
}
