"""Analyze concurrent CUPTI activity; intervals overlap, totals are not additive.

Use --list-segments to locate warmup/measured requests separated by cooldowns,
then --segment N (or explicit CUPTI --start-ns / --end-ns) for a request window.
Segments infer GPU gaps, not semantic request boundaries; verify their duration
against benchmark output. GPU busy means some activity, not SM saturation.
"""
import argparse
import collections
import datetime
import json
import gzip


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals) / 1e6


def kernel_coverage(kernels, start, end):
    # Sweep overlapping streams; identical names may occur concurrently.
    events = []
    for index, row in enumerate(kernels):
        if min(end, row['end']) <= max(start, row['start']):
            continue
        events.extend([(max(start, row['start']), 1, index), (min(end, row['end']), -1, index)])
    active = set()
    last = start
    exclusive, covered = collections.Counter(), collections.Counter()
    for timestamp, change, index in sorted(events):
        if active:
            names = {kernels[i]['name'] for i in active}
            if len(names) == 1:
                exclusive[next(iter(names))] += timestamp - last
            for name in names:
                covered[name] += timestamp - last
        if change == 1:
            active.add(index)
        else:
            active.remove(index)
        last = timestamp
    geometries = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for row in kernels:
        key = (tuple(row['grid']), tuple(row['block']), row['registers'], row['sharedBytes'])
        totals = geometries[row['name']][key]
        totals[0] += 1
        totals[1] += (min(end, row['end']) - max(start, row['start'])) / 1e6
    return sorted([dict(name=name, exclusiveMs=exclusive[name] / 1e6, unionMs=covered[name] / 1e6,
        geometries=[dict(grid=key[0], block=key[1], registers=key[2], sharedBytes=key[3], count=value[0], summedMs=value[1])
                    for key, value in sorted(values.items(), key=lambda pair: -pair[1][1])[:8]])
                   for name, values in geometries.items()], key=lambda row: -row['exclusiveMs'])


def analyze(records, start, end):
    rows = [r for r in records if r.get('end', 0) > start and r.get('start', end) < end]
    kernels = [r for r in rows if r['kind'] == 'kernel']
    copies = [r for r in rows if r['kind'] == 'memcpy']
    apis = [r for r in rows if r['kind'] in ('driver', 'runtime')]
    intervals = lambda values: union((max(start, r['start']), min(end, r['end'])) for r in values)
    busy = intervals(kernels + copies)
    compute = intervals(kernels)
    by_name = collections.defaultdict(lambda: [0, 0])
    for r in kernels:
        by_name[r['name']][0] += 1
        by_name[r['name']][1] += (min(end, r['end']) - max(start, r['start'])) / 1e6
    api_names = collections.defaultdict(lambda: [0, 0])
    for r in apis:
        api_names[r['name']][0] += 1
        api_names[r['name']][1] += (min(end, r['end']) - max(start, r['start'])) / 1e6
    correlations = {(r['kind'], r['correlation']): r for r in records if r['kind'] in ('driver', 'runtime')}
    starts = {r['start']: r for r in kernels + copies}
    gaps = []
    categories = collections.Counter()
    for left, right in zip(busy, busy[1:]):
        gap_start, gap_end = left[1], right[0]
        next_gpu = starts.get(gap_end)
        api = None if next_gpu is None else correlations.get(('driver', next_gpu['correlation'])) or correlations.get(('runtime', next_gpu['correlation']))
        category = 'uncorrelated'
        if api:
            category = 'API started during GPU idle' if api['start'] >= gap_start else 'API began before GPU idle'
        milliseconds = (gap_end-gap_start)/1e6
        categories[category] += milliseconds
        gaps.append(dict(start=gap_start, end=gap_end, milliseconds=milliseconds, category=category,
                         nextActivity=None if next_gpu is None else next_gpu.get('name', next_gpu['kind']),
                         api=None if api is None else dict(name=api['name'], start=api['start'], end=api['end'])))
    return dict(start=start, end=end, windowMs=(end-start)/1e6,
                kernels=len(kernels), copies=len(copies), computeUnionMs=duration(compute),
                gpuActivityUnionMs=duration(busy), gpuIdleMs=(end-start)/1e6-duration(busy),
                apiUnionMs=duration(intervals(apis)),
                synchronizationApiUnionMs=duration(intervals([r for r in apis if 'Synchronize' in r['name']])),
                streamCount=len({(r['device'], r['context'], r['stream']) for r in kernels}),
                kernelCoverage=kernel_coverage(kernels, start, end),
                idleGapCategoriesMs=dict(categories),
                largestIdleGaps=sorted(gaps, key=lambda r: -r['milliseconds'])[:30],
                topKernelsBySummedMs=sorted([dict(name=n, count=c, summedMs=t) for n, (c, t) in by_name.items()], key=lambda r: -r['summedMs'])[:30],
                topApisBySummedMs=sorted([dict(name=n, count=c, summedMs=t) for n, (c, t) in api_names.items()], key=lambda r: -r['summedMs'])[:30])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--list-segments', action='store_true')
    parser.add_argument('--segment', type=int)
    parser.add_argument('--gap-ms', type=float, default=250)
    parser.add_argument('--benchmark', help='Effect direct JSONL, timestamped immediately after each measured run')
    parser.add_argument('--benchmark-row', type=int, default=0)
    parser.add_argument('--start-ns', type=int)
    parser.add_argument('--end-ns', type=int)
    args = parser.parse_args()
    opener = gzip.open if args.path.endswith('.gz') else open
    with opener(args.path, 'rt') as source:
        records = [json.loads(line) for line in source if line.strip()]
    summary = [r for r in records if r['kind'] == 'summary']
    if len(summary) != 1 or any(summary[0][key] for key in ('droppedRecords', 'omittedRecords', 'errors')):
        raise SystemExit('Incomplete or invalid capture: ' + repr(summary))
    gpu = sorted((r['start'], r['end']) for r in records if r['kind'] in ('kernel', 'memcpy') and r['end'] > r['start'] > 0)
    if not gpu:
        raise SystemExit('No valid GPU records')
    segments = []
    for start, end in gpu:
        if segments and start - segments[-1][1] < args.gap_ms * 1e6:
            segments[-1][1] = max(segments[-1][1], end)
        else:
            segments.append([start, end])
    if args.list_segments:
        print(json.dumps([dict(segment=i, start=s, end=e, milliseconds=(e-s)/1e6) for i, (s,e) in enumerate(segments)], indent=2))
        return
    start, end = segments[args.segment] if args.segment is not None else (gpu[0][0], max(e for _, e in gpu))
    if args.benchmark:
        with open(args.benchmark) as source:
            benchmark = [json.loads(line) for line in source if line.strip()][args.benchmark_row]
        metadata = next(r for r in records if r['kind'] == 'metadata')
        wall = datetime.datetime.fromisoformat(benchmark['timestamp'].replace('Z', '+00:00'))
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        delta = wall - epoch
        wall_ns = (delta.days * 86400 + delta.seconds) * 1000000000 + delta.microseconds * 1000
        if 'measurementWindowWallNs' in benchmark:
            start, end = (int(value) + metadata['timestamp'] - metadata['wallNs']
                          for value in benchmark['measurementWindowWallNs'])
        else:
            end = wall_ns + 1000000 + metadata['timestamp'] - metadata['wallNs']
            start = end - round(benchmark['elapsedMilliseconds'] * 1000000)
    start = args.start_ns if args.start_ns is not None else start
    end = args.end_ns if args.end_ns is not None else end
    report = analyze(records, start, end)
    report['captureSummary'] = summary[0]
    if args.benchmark:
        report['windowSource'] = ('Explicit wall-clock timestamps immediately outside the unchanged timed request; mapped to this process CUPTI clock using initialization metadata.'
            if 'measurementWindowWallNs' in benchmark else
            'Benchmark completion wall timestamp plus 1ms minus request duration; millisecond timestamp rounding and post-request bookkeeping introduce boundary uncertainty.')
    report['caveat'] = 'Kernel/API sums overlap. GPU activity union is not SM utilization. Late API start supports submission starvation but does not establish its cause. CUPTI instrumentation changes performance.'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
