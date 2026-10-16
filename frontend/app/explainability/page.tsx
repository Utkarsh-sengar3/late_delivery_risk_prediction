import { ArrowDownToLine, Lightbulb } from "lucide-react";
import { PageLayout } from "@/components/page-layout";
import { Panel, SectionHeader, StatusPill } from "@/components/ui";
import { ImportanceChart } from "@/components/charts";
import { featureImportance } from "@/lib/data";
export default function ExplainabilityPage(){
 return <PageLayout title="Explainability" subtitle="Understand which raw input features the model relied on in Phase 6.">
  <div className="grid two-col">
   <Panel><SectionHeader title="Global permutation importance" description="Validation sample · 2,000 rows"/><div className="chart-area tall"><ImportanceChart/></div><div className="alert"><Lightbulb size={16}/><div>Permutation importance measures the change in validation F1 when feature values are shuffled. It describes model reliance, not cause and effect.</div></div></Panel>
   <Panel><SectionHeader title="Top features" description="Phase 6 measured values"/>
    <div className="table-wrap"><table className="data-table"><thead><tr><th>#</th><th>Feature</th><th>Importance</th></tr></thead><tbody>{featureImportance.map((f,i)=><tr key={f.feature}><td>{String(i+1).padStart(2,"0")}</td><td>{f.feature}</td><td>{f.value.toFixed(6)}</td></tr>)}</tbody></table></div>
    <div className="alert info section-gap"><Lightbulb size={16}/><div><strong>Prediction-time audit required.</strong> Verify each feature is available at the moment the prediction is made. Remove unavailable or post-outcome fields.</div></div>
   </Panel>
  </div>
 </PageLayout>;
}
