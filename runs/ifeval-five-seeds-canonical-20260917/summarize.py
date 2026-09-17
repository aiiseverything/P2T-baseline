"""Verify and report all 25 completed, freshly generated IFEval repetitions."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import statistics
from datetime import datetime, timezone

SUITE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ifeval_five_seed_worker', SUITE / 'run_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
METRICS = ('prompt_strict', 'prompt_loose', 'inst_strict', 'inst_loose')
COMPOSITE = 'four_metric_mean'


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    manifest = json.loads((SUITE / 'experiment.json').read_text())
    for relative, expected in manifest['files_sha256'].items():
        worker.require(worker.file_hash(SUITE / relative) == expected, f'Frozen input changed: {relative}')
    records, model_summary, counts = [], [], {}
    for tag in worker.TAGS:
        job = SUITE / 'job' / tag
        state = json.loads((job / 'status.json').read_text())
        worker.require(state['state'] == 'complete' and state['evaluations'] == 5,
                       f'{tag} has not completed all five evaluations')
        worker.require((job / 'exit_code').read_text().strip() == '0', f'{tag} did not exit successfully')
        results, generations = [], []
        for seed in worker.SEEDS:
            item = worker.validate_seed_result(SUITE / 'results', tag, seed, manifest)
            results.append(item)
            records.append({'model': tag, 'seed': seed,
                **{key + '_pct': item[key] * 100 for key in METRICS},
                'mean_tokens': item['mean_tokens'], 'truncated_count': item['capped'],
                COMPOSITE + '_pct': statistics.mean(item[key] * 100 for key in METRICS)})
            generations.append(worker.read_jsonl(SUITE / 'results' / f'seed-{seed}' / tag /
                                                'generations_t1.0_n1.jsonl'))
        row = {'model': tag, 'n_evaluations': 5}
        for key in METRICS:
            values = [result[key] * 100 for result in results]
            row[key + '_mean_pct'] = statistics.mean(values)
            row[key + '_sd_pp'] = statistics.stdev(values)
            row[key + '_min_pct'] = min(values)
            row[key + '_max_pct'] = max(values)
        row['mean_tokens'] = statistics.mean(item['mean_tokens'] for item in results)
        row['truncated_count_total'] = sum(item['capped'] for item in results)
        # The four component metrics are correlated. Compute the composite per
        # seed first, then measure its actual across-seed variation.
        composites = [statistics.mean(item[key] * 100 for key in METRICS) for item in results]
        row[COMPOSITE + '_mean_pct'] = statistics.mean(composites)
        row[COMPOSITE + '_sd_pp'] = statistics.stdev(composites)
        row[COMPOSITE + '_min_pct'] = min(composites)
        row[COMPOSITE + '_max_pct'] = max(composites)
        model_summary.append(row)
        counts[tag] = {
            'distinct_response_set_hashes': len({worker.digest([row['responses'][0] for row in batch])
                                               for batch in generations}),
            'prompts_with_different_answers_across_seeds': sum(
                len({batch[i]['responses'][0] for batch in generations}) > 1 for i in range(541)),
            'completed_at': state['finished_at'],
            'validated_results': results,
        }
    worker.require(len(records) == len({(r['model'], r['seed']) for r in records}) == 25,
                   'Expected all 25 model/seed pairs exactly once')
    report = {'verified_at': datetime.now(timezone.utc).isoformat(),
        'experiment_sha256': worker.file_hash(SUITE / 'experiment.json'),
        'analysis_source_sha256': worker.file_hash(__file__),
        'n_evaluations': 25, 'n_generated_answers': 25 * 541,
        'n_instruction_checks_per_mode': 25 * 834,
        'seeds': worker.SEEDS, 'scoring_seed': 42, 'sd_ddof': 1,
        'metrics_unit': 'percentage; standard deviations in percentage points',
        'custom_metrics': {COMPOSITE: {
            'label': 'IFEval four-metric mean (project-defined)',
            'official_ifeval_metric': False,
            'weights': {key: 0.25 for key in METRICS},
            'formula': '(prompt_strict + prompt_loose + inst_strict + inst_loose) / 4',
            'seed_aggregation': 'Compute each seed composite, then five-seed mean and sample SD (ddof=1)'}},
        'individual_results': records, 'model_summary': model_summary,
        'verification': counts,
        'interpretation': 'Repeated generations of fixed trained models; no pass@5 or independent training-seed replication.'}
    worker.atomic_text(SUITE / 'results_25.csv', csv_text(records))
    worker.atomic_text(SUITE / 'model_summary.csv', csv_text(model_summary))
    worker.save(SUITE / 'summary.json', report)
    worker.save(SUITE / 'completion.json', {'state': 'complete', 'verified_at': report['verified_at'],
        'n_evaluations': 25, 'n_generated_answers': 13525, 'summary_sha256': worker.file_hash(SUITE / 'summary.json')})
    print(json.dumps({'individual_results': records, 'model_summary': model_summary,
                      'variation_checks': {tag: {k: v for k, v in value.items() if k != 'validated_results'}
                                           for tag, value in counts.items()}}, indent=2))


if __name__ == '__main__':
    main()
