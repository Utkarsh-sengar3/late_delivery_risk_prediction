 "use client";
import { useState } from "react";
import { ArrowRight, CheckCircle2, FileSpreadsheet, LoaderCircle, UploadCloud } from "lucide-react";
import { PageLayout } from "@/components/page-layout";
import { Panel, SectionHeader, StatusPill } from "@/components/ui";
import { apiPost, apiUpload, apiConfigured } from "@/lib/api";

type Prediction = { probability?: number; predicted_probability?: number; risk?: number; prediction?: number; label?: string; threshold?: number; [key:string]: unknown };
type PredictionField = { key: string; label: string; kind: "select" | "number" | "text"; options?: string[]; value?: number };
const fields: PredictionField[] = [
  {key:"Shipping Mode",label:"Shipping mode",kind:"select",options:["Standard Class","Second Class","First Class","Same Day"]},
  {key:"Type",label:"Shipment type",kind:"select",options:["DEBIT","TRANSFER","PAYMENT","CASH"]},
  {key:"Days for shipment (scheduled)",label:"Scheduled shipping days",kind:"number",value:3},
  {key:"is_express_shipping",label:"Express shipping",kind:"select",options:["0","1"]},
  {key:"shipping_pressure_index",label:"Shipping pressure index",kind:"number",value:1},
  {key:"Category Id",label:"Category ID",kind:"number",value:1},
  {key:"Order City",label:"Order city",kind:"text",value:""},
  {key:"Order State",label:"Order state",kind:"text",value:""},
  {key:"Customer State",label:"Customer state",kind:"text",value:""},
  {key:"Latitude",label:"Latitude",kind:"number",value:0},
];
export default function PredictionsPage() {
  const [tab,setTab]=useState<"single"|"batch">("single");
  const [values,setValues]=useState<Record<string,string>>({"Shipping Mode":"Standard Class","Type":"DEBIT","Days for shipment (scheduled)":"3","is_express_shipping":"0","shipping_pressure_index":"1","Category Id":"1","Order City":"","Order State":"","Customer State":"","Latitude":"0"});
  const [threshold,setThreshold]=useState("0.3790");
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState("");
  const [result,setResult]=useState<Prediction|null>(null);
  const [file,setFile]=useState<File|null>(null);
  const [batch,setBatch]=useState<any>(null);

  async function submitSingle(e:React.FormEvent) {
    e.preventDefault(); setError(""); setResult(null); setLoading(true);
    try {
      const payload:Record<string,unknown>={};
      fields.forEach(f=>payload[f.key]=f.kind==="number"?Number(values[f.key]):f.key==="is_express_shipping"?Number(values[f.key]):values[f.key]);
      const res=await apiPost<Prediction>("/predict",{features:payload,threshold:Number(threshold)});
      const probability=Number(res.probability ?? res.predicted_probability);
      const normalized={...res,probability,threshold:Number(threshold),label:res.label ?? (probability>=Number(threshold)?"High risk":"Not flagged")};
      setResult(normalized);
      const history=JSON.parse(localStorage.getItem("ldr-history")||"[]");
      history.unshift({time:new Date().toLocaleString(),probability,threshold:Number(threshold),label:normalized.label,source:"Single prediction"});
      localStorage.setItem("ldr-history",JSON.stringify(history.slice(0,200)));
    } catch(err) {setError(err instanceof Error?err.message:"Prediction failed.");}
    finally {setLoading(false);}
  }

  async function submitBatch() {
    if(!file)return;setError("");setBatch(null);setLoading(true);
    try {
      const res=await apiUpload<any>("/predict/batch",file);setBatch(res);
      const rows=res.predictions ?? res.results ?? [];
      const history=JSON.parse(localStorage.getItem("ldr-history")||"[]");
      history.unshift({time:new Date().toLocaleString(),probability:null,threshold:Number(threshold),label:`Batch processed: ${rows.length} rows`,source:file.name});
      localStorage.setItem("ldr-history",JSON.stringify(history.slice(0,200)));
    } catch(err){setError(err instanceof Error?err.message:"Batch prediction failed.");}
    finally{setLoading(false);}
  }

  const prob=result?.probability;
  return <PageLayout title="Model prediction" subtitle="Score a single shipment or process a CSV batch with the saved model.">
    <div className="tabs-row"><button className={`tab-button ${tab==="single"?"selected":""}`} onClick={()=>setTab("single")}>Single shipment</button><button className={`tab-button ${tab==="batch"?"selected":""}`} onClick={()=>setTab("batch")}>Batch prediction <span className="tab-count">CSV</span></button></div>
    {!apiConfigured&&<div className="alert info section-gap"><CheckCircle2 size={17}/><div><strong>Frontend ready · model API not connected.</strong> Connect your FastAPI backend with <code>NEXT_PUBLIC_API_URL</code> to run real predictions. The interface will not fabricate a probability.</div></div>}
    {tab==="single"?<div className="grid prediction-layout section-gap">
      <Panel>
        <SectionHeader title="Shipment attributes" description="Enter features available at the time the risk is assessed."/>
        <form onSubmit={submitSingle}>
          <div className="form-grid">
            {fields.map(f=><div className="field" key={f.key}><label htmlFor={f.key}>{f.label}</label>
              {f.kind==="select"?<select id={f.key} value={values[f.key]} onChange={e=>setValues({...values,[f.key]:e.target.value})}>{f.options?.map(o=><option key={o} value={o}>{o}</option>)}</select>:
              <input id={f.key} type={f.kind==="number"?"number":"text"} step={f.kind==="number"?"any":undefined} value={values[f.key]??""} onChange={e=>setValues({...values,[f.key]:e.target.value})} placeholder={f.kind==="text"?"Enter value":"0"} required={f.kind!=="text"}/>}
              <span className="field-help">{f.key}</span>
            </div>)}
          </div>
          <div className="field"><label>Decision threshold</label><input type="number" min="0" max="1" step=".001" value={threshold} onChange={e=>setThreshold(e.target.value)}/><span className="field-help">Default: Phase 5 selected threshold (0.3790). This changes the classification cut-off.</span></div>
          <button className="button primary full-width" type="submit" disabled={loading||!apiConfigured}>{loading?<><LoaderCircle size={15} className="spin"/> Calculating…</>:<>Calculate risk <ArrowRight size={15}/></>}</button>
        </form>
        {error&&<div className="alert error section-gap">{error}</div>}
      </Panel>
      <Panel className="result-panel">
        <SectionHeader title="Prediction result" description="Model output appears here after a successful API response."/>
        {result&&typeof prob==="number"&&Number.isFinite(prob)?<div className="result-card">
          <div className="result-overline">ESTIMATED LATE-DELIVERY PROBABILITY</div><div className="result-score">{(prob*100).toFixed(1)}<span>%</span></div>
          <div className="result-subtitle">Compared with threshold {(Number(threshold)*100).toFixed(1)}%</div>
          <div className="progress-track"><div className="progress-fill" style={{width:`${Math.max(0,Math.min(100,prob*100))}%`}}/></div>
          <StatusPill tone={prob>=Number(threshold)?"danger":"good"}>{String(result.label)}</StatusPill>
          <div className="result-foot"><span>Decision threshold</span><strong>{Number(threshold).toFixed(4)}</strong></div>
        </div>:<div className="result-placeholder"><div className="result-orbit"><TargetIcon/></div><h3>Ready when you are</h3><p>Submit shipment attributes to receive the model's estimated probability and risk classification.</p><div className="result-foot"><span>Model family</span><strong>Random Forest</strong></div><div className="result-foot"><span>Selected threshold</span><strong>0.3790</strong></div></div>}
        <div className="alert section-gap"><span>i</span><div>Predictions are decision support, not delivery guarantees. Confirm all inputs are available at prediction time.</div></div>
      </Panel>
    </div>:<div className="grid two-col section-gap">
      <Panel><SectionHeader title="Upload shipment data" description="CSV must contain all raw input columns expected by the backend."/>
        <label className="upload-drop"><UploadCloud size={28}/><strong>{file?file.name:"Choose a CSV file"}</strong><span>{file?`${(file.size/1024).toFixed(1)} KB`:"Drag and drop or browse from your computer"}</span><input type="file" accept=".csv,text/csv" onChange={e=>setFile(e.target.files?.[0]||null)}/></label>
        <div className="alert section-gap"><FileSpreadsheet size={16}/><div><strong>Before uploading</strong><br/>Use the same raw feature names as the Phase 2 preprocessor. Avoid personal information and post-outcome fields unless explicitly required by the validated model.</div></div>
        <button className="button primary full-width section-gap" onClick={submitBatch} disabled={!file||loading||!apiConfigured}>{loading?"Processing…":"Validate & predict batch"}</button>
        {error&&<div className="alert error section-gap">{error}</div>}
      </Panel>
      <Panel><SectionHeader title="Batch results" description="Actual results returned by the prediction API."/>
        {batch?<><pre className="json-output">{JSON.stringify(batch,null,2).slice(0,12000)}</pre><button className="button" onClick={()=>{const blob=new Blob([JSON.stringify(batch,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="batch-predictions.json";a.click();URL.revokeObjectURL(url);}}>Download results</button></>:<div className="result-placeholder"><div className="result-orbit"><FileSpreadsheet size={22}/></div><h3>No batch results yet</h3><p>Upload a CSV and run the prediction endpoint to see scored rows here.</p></div>}
      </Panel>
    </div>}
  </PageLayout>;
}
function TargetIcon(){return <CheckCircle2 size={27}/>}
