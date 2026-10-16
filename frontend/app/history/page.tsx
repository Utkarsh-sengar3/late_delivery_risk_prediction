 "use client";
import { useEffect,useState } from "react";
import { Download, Search, Trash2 } from "lucide-react";
import { PageLayout } from "@/components/page-layout";
import { EmptyState, Panel, SectionHeader, StatusPill } from "@/components/ui";
type Row={time:string;probability:number|null;threshold:number;label:string;source:string};
export default function HistoryPage(){
 const [rows,setRows]=useState<Row[]>([]);const [query,setQuery]=useState("");
 useEffect(()=>{try{setRows(JSON.parse(localStorage.getItem("ldr-history")||"[]"));}catch{}},[]);
 const filtered=rows.filter(r=>`${r.label} ${r.source} ${r.time}`.toLowerCase().includes(query.toLowerCase()));
 function exportCsv(){const headers=["time","probability","threshold","label","source"];const csv=[headers.join(","),...filtered.map(r=>headers.map(k=>JSON.stringify((r as any)[k]??"")).join(","))].join("\n");const url=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));const a=document.createElement("a");a.href=url;a.download="prediction-history.csv";a.click();URL.revokeObjectURL(url);}
 function clear(){localStorage.removeItem("ldr-history");setRows([]);}
 return <PageLayout title="Prediction history" subtitle="Review recent predictions stored in this browser.">
  <Panel><SectionHeader title="Recent activity" description={`${filtered.length} records in this browser`} action={<div className="button-row"><button className="button" onClick={exportCsv} disabled={!filtered.length}><Download size={14}/> Export CSV</button><button className="button" onClick={clear} disabled={!rows.length}><Trash2 size={14}/> Clear history</button></div>}/>
   <div className="table-toolbar"><div className="search-input"><Search size={15}/><input placeholder="Search predictions…" value={query} onChange={e=>setQuery(e.target.value)}/></div><span className="muted-text">Browser-local storage</span></div>
   {filtered.length?<div className="table-wrap"><table className="data-table"><thead><tr><th>Timestamp</th><th>Source</th><th>Probability</th><th>Threshold</th><th>Decision</th></tr></thead><tbody>{filtered.map((r,i)=><tr key={`${r.time}-${i}`}><td>{r.time}</td><td>{r.source}</td><td>{r.probability==null?"—":`${(r.probability*100).toFixed(1)}%`}</td><td>{r.threshold.toFixed(4)}</td><td><StatusPill tone={r.label.toLowerCase().includes("risk")?"danger":"neutral"}>{r.label}</StatusPill></td></tr>)}</tbody></table></div>:<EmptyState title="No predictions recorded" description="When the backend is connected and you run a prediction, its summary will appear here. History is stored only in this browser."/>}
  </Panel>
 </PageLayout>;
}
