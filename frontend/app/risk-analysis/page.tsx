 "use client";
import { useState } from "react";
import { Filter, RefreshCw } from "lucide-react";
import { PageLayout } from "@/components/page-layout";
import { EmptyState, Panel, SectionHeader } from "@/components/ui";
import { apiConfigured, apiGet } from "@/lib/api";
export default function RiskAnalysisPage(){
 const [segment,setSegment]=useState("Shipping Mode");const [data,setData]=useState<any>(null);const [error,setError]=useState("");
 async function run(){setError("");try{setData(await apiGet(`/analytics/segments?group_by=${encodeURIComponent(segment)}`));}catch(e){setError(e instanceof Error?e.message:"Could not load segment analysis.");}}
 return <PageLayout title="Risk analysis" subtitle="Investigate how model risk differs across operational segments.">
  <Panel><SectionHeader title="Segment explorer" description="Choose a segment to request an analysis from the connected API" action={<button className="button primary" onClick={run} disabled={!apiConfigured}><Filter size={14}/> Apply filters</button>}/>
   <div className="field" style={{maxWidth:320}}><label>Group shipments by</label><select value={segment} onChange={e=>setSegment(e.target.value)}>{["Shipping Mode","Type","Order State","Customer State","Category Name","Market"].map(x=><option key={x}>{x}</option>)}</select></div>
   {data?<pre className="json-output">{JSON.stringify(data,null,2).slice(0,12000)}</pre>:<EmptyState title="No segment results yet" description="Connect the analytics endpoint to display actual observed late rates and model-predicted probabilities by segment."/>}
   {error&&<div className="alert error section-gap">{error}</div>}
  </Panel>
  <div className="alert section-gap"><Filter size={16}/><div><strong>Keep two measures separate.</strong> Observed late-delivery rate comes from labelled historical data. Predicted risk is the model's estimate. Do not interpret either as proof that a location or shipping mode causes delays.</div></div>
 </PageLayout>;
}
