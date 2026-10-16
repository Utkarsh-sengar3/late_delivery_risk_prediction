 "use client";
import { useState } from "react";
import { Activity, AlertTriangle, CheckCircle2, Download, Gauge, Target } from "lucide-react";
import { PageLayout } from "@/components/page-layout";
import { MetricCard, Panel, SectionHeader, EmptyState, StatusPill } from "@/components/ui";
import { modelMetrics } from "@/lib/data";
import { apiGet, apiConfigured } from "@/lib/api";

export default function PerformancePage(){
 const [live,setLive]=useState<any>(null);const [error,setError]=useState("");
 async function refresh(){setError("");try{setLive(await apiGet("/model/performance"));}catch(e){setError(e instanceof Error?e.message:"Could not load performance.");}}
 const metrics=[{label:"Accuracy",value:"63.65%",note:"Held-out test · Phase 5",icon:Gauge,tone:"blue" as const},{label:"Precision",value:"62.33%",note:"Held-out test · Phase 5",icon:Target,tone:"teal" as const},{label:"Recall",value:"85.20%",note:"Held-out test · Phase 5",icon:Activity,tone:"amber" as const},{label:"F1 score",value:"71.99%",note:"Held-out test · Phase 5",icon:CheckCircle2,tone:"violet" as const}];
 return <PageLayout title="Performance & errors" subtitle="Understand the model's measured performance, trade-offs, and failure modes." actions={<button className="button" onClick={refresh} disabled={!apiConfigured}><Activity size={14}/> Refresh test evaluation</button>}>
  <div className="metrics-grid grid">{metrics.map(m=><MetricCard key={m.label} {...m}/>)}</div>
  <div className="grid two-col">
   <Panel><SectionHeader title="Evaluation summary" description="Recorded Phase 5 held-out test metrics"/>
    <div className="metric-list">{modelMetrics.map(m=><div className="metric-list-row" key={m.label}><span>{m.label}</span><strong>{m.value}</strong><span className="metric-list-note">{m.note}</span></div>)}</div>
   </Panel>
   <Panel><SectionHeader title="Error analysis" description="Counts require actual test predictions"/>
    {live?<pre className="json-output">{JSON.stringify(live,null,2).slice(0,10000)}</pre>:<EmptyState title="Confusion matrix not connected" description="The recorded report contains aggregate metrics, but the exact TP/TN/FP/FN counts are not available in the current frontend package. Connect /model/performance to calculate them from the saved model and held-out test split."/>}
   </Panel>
  </div>
  {error&&<div className="alert error section-gap">{error}</div>}
  <div className="alert info section-gap"><AlertTriangle size={17}/><div><strong>Evaluation discipline.</strong> Threshold 0.3790 was selected on validation data and then evaluated on the held-out test set. The test set should not be reused to tune the production threshold.</div></div>
  <Panel className="section-gap"><SectionHeader title="What the metrics mean" description="Quick interpretation for model review"/>
   <div className="grid three-col"><div className="definition-card"><strong>Precision</strong><p>Among shipments flagged as late risk, how many were actually late?</p></div><div className="definition-card"><strong>Recall</strong><p>Among actually late shipments, how many did the model flag?</p></div><div className="definition-card"><strong>F1 score</strong><p>Harmonic mean of precision and recall at the selected threshold.</p></div></div>
  </Panel>
 </PageLayout>;
}
