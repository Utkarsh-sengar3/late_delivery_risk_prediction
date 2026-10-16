 "use client";
import { useState } from "react";
import { ArrowDownToLine, ChartNoAxesCombined, RefreshCw } from "lucide-react";
import { PageLayout } from "@/components/page-layout";
import { EmptyState, Panel, SectionHeader, StatusPill } from "@/components/ui";
import { apiGet, apiConfigured } from "@/lib/api";
import { ImportanceChart } from "@/components/charts";

export default function AnalyticsPage(){
  const [payload,setPayload]=useState<any>(null);const [error,setError]=useState("");const [loading,setLoading]=useState(false);
  async function load(){setLoading(true);setError("");try{setPayload(await apiGet("/analytics/predictions"));}catch(e){setError(e instanceof Error?e.message:"Could not load analytics.");}finally{setLoading(false);}}
  return <PageLayout title="Prediction analytics" subtitle="Explore prediction patterns and shipment segments without confusing model scores with actual outcomes." actions={<button className="button" onClick={load} disabled={loading||!apiConfigured}><RefreshCw size={14}/>{loading?"Refreshing":"Refresh data"}</button>}>
    <div className="grid two-col">
      <Panel><SectionHeader title="Global feature importance" description="Phase 6 · raw-feature permutation importance"/><div className="chart-area tall"><ImportanceChart/></div><div className="chart-footnote">Higher values indicate stronger impact on validation F1 when the feature is shuffled. This is not causal evidence.</div></Panel>
      <Panel><SectionHeader title="Prediction series" description="Risk distribution from real model responses"/>{payload?<pre className="json-output">{JSON.stringify(payload,null,2).slice(0,8000)}</pre>:<EmptyState title="No prediction series loaded" description="Connect the API and use Refresh data to load actual prediction analytics."/>}</Panel>
    </div>
    <div className="grid two-col section-gap">
      <Panel><SectionHeader title="Segment comparison" description="Compare risk by shipping mode, region, or category"/><EmptyState title="Waiting for analytics endpoint" description="The frontend does not invent regional rates. Add a real /analytics/predictions response to populate this panel."/></Panel>
      <Panel><SectionHeader title="Cluster exploration" description="PCA / K-Means shipment segments"/><EmptyState title="Cluster data not connected" description="Cluster points should be computed from the project dataset or returned by the backend, not fabricated in the browser."/></Panel>
    </div>
    {error&&<div className="alert error section-gap">{error}</div>}
  </PageLayout>;
}
