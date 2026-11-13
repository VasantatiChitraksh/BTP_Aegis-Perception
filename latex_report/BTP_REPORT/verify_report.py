#!/usr/bin/env python3
"""Verify report evidence, immutable inputs, rendered bounds and PDF links."""
import csv
import bisect
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent

def rows(name):
    return list(csv.DictReader((ROOT/'evidence'/name).open()))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_histograms(checks, pdf):
    """Validate compact observation snapshots, labels and available raw sources.

    This uses only the standard library and PyMuPDF so that the standalone
    source bundle does not need NumPy or the original, larger experiment logs.
    When those logs are present, every bin count is independently reconstructed.
    """
    evidence=ROOT/'evidence'
    restoration=json.loads((evidence/'restoration_gain_histogram.json').read_text())
    latency=json.loads((evidence/'nano_latency_histogram.json').read_text())
    provenance=json.loads((evidence/'distribution_charts_provenance.json').read_text())
    for name,entry in provenance['figures'].items():
        checks['histogram_snapshot_hash_'+name]=digest(ROOT/entry['snapshot'])==entry['sha256']
        checks['histogram_figure_hash_'+name]=digest(ROOT/'figures'/name)==entry['figure_sha256']
    checks['histogram_generator_hash']=digest(ROOT/provenance['generator'])==provenance['generator_sha256']

    def valid_series(s, width):
        edges=s['bin_edges']; counts=s['bin_counts']; n=s['n']
        numbers=edges+[s[k] for k in ('median','p95','min','max')]
        return (isinstance(n,int) and n>0 and len(edges)==len(counts)+1
            and all(math.isfinite(v) for v in numbers)
            and all(isinstance(v,int) and v>=0 for v in counts)
            and all(math.isclose(b-a,width,abs_tol=1e-10) for a,b in zip(edges,edges[1:]))
            and sum(counts)==n and math.isclose(sum(100*v/n for v in counts),100,abs_tol=1e-10)
            and edges[0]<=s['min']<=s['median']<=s['p95']<=s['max']<=edges[-1])

    r_index={(s['weather'],s['model']):s for s in restoration['series']}
    expected={(w,m) for w in ('rain','fog','snow','smoke') for m in ('restormer','ramit')}
    checks['restoration_histogram_series']=set(r_index)==expected and len(restoration['series'])==len(expected)
    tests={v['weather']:int(v['pairs']) for v in rows('dataset_split.csv') if v['split']=='test'}
    checks['restoration_histogram_image_counts']=all(s['n']==tests[s['weather']] for s in restoration['series'])
    checks['restoration_histogram_normalization']=restoration['bin_width_db']==2 and all(valid_series(s,2) for s in restoration['series'])
    checks['restoration_histogram_gain_fractions']=all(0<=s['positive_gain_percent']<=100 for s in restoration['series'])
    with pdf.open(ROOT/'figures/restoration_gain_histograms.pdf') as chart:
        labels='\n'.join(p.get_text() for p in chart)
    checks['restoration_histogram_median_labels']=all(f"{s['median']:.2f} dB" in labels for s in restoration['series'])

    l_index={(s['mode'],s['weather']):s for s in latency['series']}
    expected={(m,w) for m in ('raw','restored') for w in ('fog','rain','snow')}
    checks['nano_histogram_series']=set(l_index)==expected and len(latency['series'])==len(expected)
    checks['nano_histogram_normalization']=latency['bin_width_ms']==20 and all(valid_series(s,20) for s in latency['series'])
    performance={(v['mode'],v['case']):v for v in rows('S2_dawn_performance.csv')}
    checks['nano_histogram_matches_saved_runs']=all(
        s['n']==int(performance[k]['measured_frames']) and s['unique_images']==int(performance[k]['unique_frames'])
        and math.isclose(s['median'],float(performance[k]['pipeline_median_ms']),abs_tol=1e-8)
        and math.isclose(s['p95'],float(performance[k]['pipeline_p95_ms']),abs_tol=1e-8)
        for k,s in l_index.items())
    checks['nano_histogram_recorded_scope']=(latency['repeat']==1 and latency['warmup_excluded']==5
        and latency['restorer_shape_hw']==[128,128] and latency['minimum_seconds_per_case']==30
        and all(float(v['seconds'])>=30 for v in performance.values()))
    with pdf.open(ROOT/'figures/nano_pipeline_latency_histograms.pdf') as chart:
        labels='\n'.join(p.get_text() for p in chart)
    checks['nano_histogram_median_p95_labels']=all(
        f"Median {s['median']:.2f} ms" in labels and f"P95 {s['p95']:.2f} ms" in labels
        for s in latency['series'])

    def matches_observations(s, values):
        values=sorted(values); counts=[0]*len(s['bin_counts']); edges=s['bin_edges']
        for value in values:
            i=bisect.bisect_right(edges,value)-1
            if value==edges[-1]: i=len(counts)-1
            if not 0<=i<len(counts): return False
            counts[i]+=1
        position=(len(values)-1)*.95
        lo=math.floor(position); hi=math.ceil(position)
        p95=values[lo]+(values[hi]-values[lo])*(position-lo)
        return (len(values)==s['n'] and counts==s['bin_counts']
            and math.isclose(statistics.median(values),s['median'],abs_tol=1e-9)
            and math.isclose(p95,s['p95'],abs_tol=1e-9)
            and values[0]==s['min'] and values[-1]==s['max'])

    repo=ROOT.parents[1]
    verified_sources=0
    source=repo/restoration['source']['path']
    if source.exists():
        checks['restoration_histogram_original_source_hash']=digest(source)==restoration['source']['sha256']
        with gzip.open(source,'rt') as handle: original=list(csv.DictReader(handle))
        for (weather,model),s in r_index.items():
            values=[float(v['psnr_gain_db']) for v in original if v['weather']==weather and v['model']==model]
            checks['restoration_histogram_original_bins_'+weather+'_'+model]=matches_observations(s,values)
        verified_sources+=1
    identity=latency['identity_source']; identity_path=repo/identity['path']
    if identity_path.exists():
        checks['nano_histogram_original_identity_hash']=digest(identity_path)==identity['sha256']
        verified_sources+=1
    for s in latency['series']:
        source=repo/s['source']['path']
        if source.exists():
            key=s['mode']+'_'+s['weather']
            checks['nano_histogram_original_source_hash_'+key]=digest(source)==s['source']['sha256']
            original=json.loads(source.read_text())
            checks['nano_histogram_original_bins_'+key]=matches_observations(s,[v['pipeline_ms'] for v in original['frames']])
            verified_sources+=1
    return {'restoration_observations':sum(s['n'] for s in restoration['series']),
        'nano_timing_observations':sum(s['n'] for s in latency['series']),
        'original_source_files_checked':verified_sources,
        'normalization':'Every weather/model or weather/path histogram sums to 100%; repeated Nano frames are not independent images.'}

def main():
    import pymupdf as pdf
    checks={}
    protected=json.loads((ROOT/'evidence/immutable_inputs.json').read_text())
    checks['protected_inputs']=all((ROOT/p).exists() and digest(ROOT/p)==h for p,h in protected.items())
    # Exact table rounding from saved six-model results.
    by={(v['model'],v['weather']):v for v in rows('six_models.csv')}
    text=(ROOT/'sections/06_restoration.tex').read_text()
    for model in ('Restormer','PromptIR','TransWeather','RAMiT','LiteWeather v2','Attention Pix2Pix'):
        line=next(l for l in text.splitlines() if l.startswith(model+' &'))
        clean=re.sub(r'\\textbf\{([^}]+)\}',r'\1',line)
        cells=clean.split('&')
        macro=by[model,'macro_weather']
        expected=[f"{int(macro['parameters'])/1e6:.2f}",str(macro['selected_epoch'])]
        expected += [f"{float(by[model,w]['psnr_db']):.2f}" for w in ('rain','fog','snow','smoke')]
        expected += [f"{float(macro['psnr_db']):.2f}",f"{100*float(macro['ssim']):.2f}"]
        actual=[re.sub(r'\\.*','',v).strip() for v in cells[1:]]
        checks['table_'+model]=actual==expected
    val=rows('joint_validation.csv')
    checks['eight_continuation_arms_rejected']=len(val)==8 and all(v['passes_screen']=='False' for v in val)
    joint={v['model']:v for v in rows('joint_test.csv')}
    checks['continuation_gain']=round(float(joint['control']['test_psnr_db'])-float(joint['original']['test_psnr_db']),4)==0.1190
    # Object accounting independent of rounded presentation values.
    accuracy=rows('S1_accuracy.csv')
    for v in accuracy:
        checks['object_accounting_'+v['mode']+'_'+v['weather']]=int(v['tp'])+int(v['fn'])==int(v['ground_truth_objects'])
    allrows={v['mode']:v for v in accuracy if v['weather']=='all'}
    checks['nano_72_images_541_objects']=all(int(v['images'])==72 and int(v['ground_truth_objects'])==541 for v in allrows.values())
    checks['nano_ap_drop']=round(100*(float(allrows['raw']['map50_95'])-float(allrows['restored']['map50_95'])),2)==15.08
    memory=(ROOT/'tables/nano_memory.tex').read_text()
    for v in rows('S2_dawn_performance.csv'):
        line='{} & {} & {:.0f} & {:.0f} & {:.1f}'.format(v['case'].title(),'YOLO' if v['mode']=='raw' else 'Cascade',float(v['ram_peak_mb']),float(v['swap_peak_mb']),float(v['gpu_mean_percent']))
        checks['memory_'+v['case']+'_'+v['mode']]=line in memory
        checks['energy_'+v['case']+'_'+v['mode']]=abs(float(v['input_energy_j'])/int(v['measured_frames'])-float(v['input_j_per_frame']))<1e-10
    doc=pdf.open(ROOT/'BTP_Report.pdf')
    # Check the new visual labels against full-precision numerical snapshots.
    models=('Restormer','PromptIR','TransWeather','RAMiT','LiteWeather v2','Attention Pix2Pix')
    for name,key,mult in [('weather_psnr','psnr_db',1),('weather_ssim','ssim',100)]:
        chart=pdf.open(ROOT/'figures'/f'{name}.pdf')[0].get_text()
        checks[name+'_rounding']=all(f"{mult*float(by[m,w][key]):.2f}" in chart for m in models for w in ('rain','fog','snow','smoke'))
    fixtures=json.loads((ROOT/'evidence/S13_nano_restorer_validation.json').read_text())
    fixture_table=(ROOT/'tables/nano_engine_agreement.tex').read_text()
    checks['fixture_scaled_rounding']=all('{} & {:.2f} & {:.2f} & {:+.2f}'.format(v['sample'].title(),v['rmse']*1e4,v['max_abs_error']*1e3,v['psnr_delta_db']*1e3) in fixture_table for v in fixtures['rows'])
    gain_chart=pdf.open(ROOT/'figures/continuation_millidb.pdf')[0].get_text()
    checks['continuation_chart_labels']=all(f"{1000*(float(v['test_psnr_db'])-float(joint['control']['test_psnr_db'])):+.2f}" in gain_chart for k,v in joint.items() if k not in ('control','original'))
    appendix_pages=[p for p in doc if p.get_label().startswith('S')]
    checks['appendix_precision_two_decimals']=bool(appendix_pages) and not any(re.search(r'(?<!\w)\d+\.\d{3,}',p.get_text()) for p in appendix_pages)
    checks['supplementary_material_visible_heading']=bool(appendix_pages) and any(
        'Supplementary Material' in ' '.join(s['text'] for line in block.get('lines',[]) for s in line['spans'])
        and any(s['size']>=16 for line in block.get('lines',[]) for s in line['spans'])
        for block in appendix_pages[0].get_text('dict')['blocks'] if block['type']==0)
    histogram_details=verify_histograms(checks,pdf)
    chart_bounds=[]
    project=json.loads((ROOT/'evidence/project_visuals.json').read_text())
    results=json.loads((ROOT/'evidence/new_visuals.json').read_text())
    figure_names=set(project['figures'])|set(results['figures'])|{'cover_pipeline.pdf'}
    distributions=json.loads((ROOT/'evidence/distribution_charts_provenance.json').read_text())
    galleries=json.loads((ROOT/'evidence/comparison_galleries.json').read_text())
    gallery_inputs=ROOT/'figures/comparison_inputs'
    gallery_manifest=json.loads((gallery_inputs/'provenance.json').read_text())
    checks['gallery_saved_inputs_unchanged']=all(
        digest(gallery_inputs/v['file'])==v['sha256'] for v in gallery_manifest['inputs'])
    checks['gallery_twenty_four_saved_panels']=sum(v['panels'] for v in galleries['figures'])==24
    matched_rows=[]
    for source,details in gallery_manifest['sources'].items():
        for weather in ('fog','rain','snow','smoke'):
            records=[json.loads((gallery_inputs/source/weather/m/'result.json').read_text())
                for m in ('direct','resize_128','restored_128')]
            source_hash=digest(gallery_inputs/source/weather/'weather_source.png')
            matched_rows.append(all(v['weather_source_sha256']==source_hash for v in records)
                and all(v['input_shape']==[1,3,128,128] for v in records[1:])
                and all(v['matches']['tp']+v['matches']['fn']==len(details['gt']) for v in records))
    checks['gallery_matched_source_geometry_and_counts']=all(matched_rows) and len(matched_rows)==8
    figure_names.update(distributions['figures'])
    figure_names.update(entry['figure'] for entry in galleries['figures'])
    figure_names.update(('work_timeline.pdf','weather_examples_mono.pdf','dawn_operating_points_spaced.pdf','edge_acceptance_current.pdf'))
    for asset in [ROOT/'figures'/name for name in sorted(figure_names)]:
        with pdf.open(asset) as chart:
            for page in chart:
                for w in page.get_text('words'):
                    if w[0]<-.1 or w[1]<-.1 or w[2]>page.rect.width+.1 or w[3]>page.rect.height+.1:
                        chart_bounds.append([asset.name,page.number+1,w[4]])
                for image in page.get_image_info():
                    x0,y0,x1,y1=image['bbox']
                    if x0<-.1 or y0<-.1 or x1>page.rect.width+.1 or y1>page.rect.height+.1:
                        chart_bounds.append([asset.name,page.number+1,'image',list(image['bbox'])])
    checks['new_chart_text_inside_canvas']=not chart_bounds
    checks['ongoing_phase_front_and_conclusion']=sum('RAMiT + YOLOv12' in ' '.join(p.get_text().split()) for p in doc)>=4
    figure_targets=re.findall(r'\\newlabel\{fig:[^}]+\}\{\{[^}]+\}\{([^}]+)\}.*?\{(figure\.[^}]+)\}',(ROOT/'.build/main.aux').read_text())
    cover=doc[0].get_text()
    checks['cover_identity']=all(v in cover for v in ('Chitraksh','CS23B054','Sudhanva','CS23B051','Rishikesh','CS23B015','Guide: Dr Vishnu Chalavadi'))
    checks['cover_has_no_phase_status']=not any(v in cover.lower() for v in ('ongoing','completed','phase 0','six experimental'))
    checks['cover_module_label']=pdf.open(ROOT/'figures/cover_pipeline.pdf')[0].get_text().strip()=='Aegis Perception module'
    hardware=json.loads((ROOT/'evidence/cover_hardware.json').read_text())
    checks['original_hardware_photos_preserved']=all(digest(ROOT/name)==value for name,value in hardware['hardware_photos'].items())
    checks['standalone_title_page']='B.Tech Project Report' in cover and 'Aegis Perception' in cover
    opening=doc[2].get_text(); work=doc[3].get_text()
    checks['opening_images_and_results_on_page_3']=all(v in opening for v in
        ('Project Overview','Degraded','Restormer','RAMiT','Clean target','Matched continuation','29.97','32.86','14.50'))
    checks['vehicle_comparison_on_page_5']=all(v in doc[4].get_text() for v in ('Matched Vehicle Detection','Matched vehicle comparison:','YOLOv8n only','Restormer + YOLOv8n'))
    checks['workflow_moved_to_page_6']='Experimental workflow.' in doc[5].get_text() and 'Experimental workflow.' not in opening
    coverage=[]
    for i in range(2,17):
        captions=re.findall(r'^(?:Figure|Table)\s+(?:S)?\d+:.*',doc[i].get_text(),flags=re.M)
        coverage.append({'physical_page':i+1,'printed_page':doc[i].get_label(),'captions':captions})
    checks['visuals_on_each_of_first_15_body_pages']=all(row['captions'] for row in coverage)
    (ROOT/'validation/visual_spread.json').write_text(json.dumps(coverage,indent=2)+'\n')
    neutral=json.loads((ROOT/'evidence/neutral_comparison.json').read_text())
    checks['neutral_comparison_source_hash']=digest(ROOT/neutral['source'])==neutral['source_sha256']
    checks['neutral_comparison_output_hash']=digest(ROOT/neutral['output'])==neutral['output_sha256']
    with pdf.open(ROOT/neutral['source']) as original, pdf.open(ROOT/neutral['output']) as updated:
        image_hashes=lambda d:sorted(hashlib.sha256(d.xref_stream(i[0])).hexdigest() for i in d[0].get_images())
        checks['weather_comparison_pixels_unchanged']=image_hashes(original)==image_hashes(updated)
    for name in ('weather_examples_mono.pdf','matched_128_vehicles.pdf','matched_128_people.pdf'):
        with pdf.open(ROOT/'figures'/name) as chart:
            spans=[s for b in chart[0].get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans']]
            checks['black_comparison_labels_'+name]=bool(spans) and all(s['color']==0 for s in spans)
    repairs=json.loads((ROOT/'evidence/chart_layout_repairs.json').read_text())
    checks['chart_layout_repairs_hashes']=all(digest(ROOT/v[k])==v[k+'_sha256'] for v in repairs['figures'] for k in ('source','output'))
    checks['timeline_and_contributions_on_page_4']=all(v in work for v in
        ('Project Timeline and Contributions','Chitraksh','Sudhanva','Rishikesh','26 Aug','13 Nov','Custom quantizer'))
    checks['retired_presentation_cards_absent']=not any(v in '\n'.join(p.get_text() for p in doc) for v in
        ('SIX COMPLETED EXPERIMENTAL PHASES','WHO DID WHAT','NANO THROUGHPUT','ONGOING EXPLORATION'))
    checks['unique_figure_destinations']=len(figure_targets)==len(set(target for _,target in figure_targets))
    checks['standalone_page_count_20_to_30']=20<=len(doc)<=30
    checks['academic_a4']=all(abs(p.rect.width-595.276)<.1 and abs(p.rect.height-841.89)<.1 for p in doc)
    bounds=[];invalid_links=[]
    for i,p in enumerate(doc):
        for w in p.get_text('words'):
            if w[0]<47 or w[2]>p.rect.width-47 or w[1]<15 or w[3]>p.rect.height-15:
                bounds.append([i+1,list(w[:5])])
        for link in p.get_links():
            if link['kind']==pdf.LINK_GOTO and not 0<=link['page']<len(doc):invalid_links.append([i+1,link])
    checks['text_inside_page_bounds']=not bounds
    checks['valid_academic_link_targets']=not invalid_links
    log=(ROOT/'.build/main.log').read_text(errors='replace')
    critical=[line for line in log.splitlines() if any(x in line for x in ('Overfull \\', 'undefined references','undefined citations','multiply defined','LaTeX Error:','Missing character:'))]
    checks['no_critical_latex_warnings']=not critical
    result={'checks':checks,'passed':all(checks.values()),'academic_pages':len(doc),'workshop_pages_included':0,'underfull_hbox_notices':log.count('Underfull \\hbox'),'out_of_bounds_words':bounds,'critical_warnings':critical,'out_of_bounds_chart_items':chart_bounds,'histogram_verification':histogram_details}
    repo=ROOT.parents[1]
    if (repo/'docs/manuscript/verify_evidence.py').exists():
        run=subprocess.run(['python3',str(repo/'docs/manuscript/verify_evidence.py')],capture_output=True,text=True,cwd=repo)
        result['manuscript_verifier']={'returncode':run.returncode,'output':run.stdout.strip()}
        assert run.returncode==0,run.stdout+run.stderr
        source=repo/'artifacts/edge_v1/nano_20261111/consolidated_report_v2'
        index=json.loads((source/'source_index.json').read_text())
        result['nano_source_index_verified']=all(digest(source/'sources'/n)==v['sha256'] for n,v in index.items())
        assert result['nano_source_index_verified']
    (ROOT/'validation/evidence_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['passed'],[k for k,v in checks.items() if not v]

if __name__=='__main__':main()
