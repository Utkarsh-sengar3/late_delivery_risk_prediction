 "use client";
import Link from "next/link";
import { Activity, ArrowRight, ArrowUpRight, Boxes, CircleGauge, Clock3, Download, PackageCheck, ShieldAlert, Target } from "lucide-react";
import { PageLayout } from "@/components/page-layout";
import { MetricCard, Panel, SectionHeader, StatusPill } from "@/components/ui";
import { ImportanceMiniChart, RiskDonut } from "@/components/charts";
import { modelMetrics, featureImportance } from "@/lib/data";

export default function DashboardPage() {
  return <PageLayout title="Overview" subtitle="A clear view of model health, shipment risk, and the work ahead."
    actions={<Link className="button primary" href="/predictions"><Target size={15}/> Run a prediction <ArrowRight size={14}/></Link>}>
    <div className="metrics-grid grid">
      <MetricCard label="Dataset records" value="180,519" note="Phase 1 audited dataset" icon={Boxes} tone="blue"/>
      <MetricCard label="Test F1 score" value="71.99%" note="Recorded Phase 5 evaluation" icon={CircleGauge} tone="teal"/>
      <MetricCard label="Test recall" value="85.20%" note="Recorded held-out test result" icon={Activity} tone="amber"/>
      <MetricCard label="Decision threshold" value="0.3790" note="Selected on validation data" icon={Target} tone="violet"/>
    </div>
    <div className="grid two-col">
      <Panel>
        <SectionHeader title="Model signal overview" description="Recorded held-out test metrics from Phase 5" action={<Link className="button ghost" href="/performance">View performance <ArrowUpRight size={13}/></Link>}/>
        <div className="chart-area"><ImportanceMiniChart/></div>
        <div className="chart-footnote">Feature importance values are from Phase 6 permutation analysis. They indicate predictive reliance, not causation.</div>
      </Panel>
      <Panel>
        <SectionHeader title="Predicted risk distribution" description="Appears after real prediction records are available"/>
        <div className="chart-area"><RiskDonut/></div>
      </Panel>
    </div>
    <div className="grid two-col section-gap">
      <Panel>
        <SectionHeader title="Model performance snapshot" description="Held-out test metrics — Phase 5"/>
        <div className="metric-list">
          {modelMetrics.map(m=><div className="metric-list-row" key={m.label}><span>{m.label}</span><strong>{m.value}</strong><span className="metric-list-note">{m.note}</span></div>)}
        </div>
        <Link href="/performance" className="text-link">Explore error analysis <ArrowRight size={14}/></Link>
      </Panel>
      <Panel>
        <SectionHeader title="Feature importance" description="Top signals identified by Phase 6"/>
        <div className="importance-list">
          {featureImportance.slice(0,5).map((f,i)=><div className="importance-row" key={f.feature}><div className="importance-rank">{String(i+1).padStart(2,"0")}</div><div className="importance-content"><div className="importance-label"><span>{f.feature}</span><b>{f.value.toFixed(5)}</b></div><div className="importance-track"><span style={{width:`${(f.value/featureImportance[0].value)*100}%`}}/></div></div></div>)}
        </div>
        <Link href="/explainability" className="text-link">Open explainability <ArrowRight size={14}/></Link>
      </Panel>
    </div>
    <div className="alert info section-gap"><ShieldAlert size={17}/><div><strong>Interpret the model responsibly.</strong> The original split was random, so future-time generalization has not been established. Validate feature availability before operational use.</div></div>
  </PageLayout>;
}
