 "use client";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area, PieChart, Pie, Cell, Legend } from "recharts";
import { featureImportance } from "@/lib/data";

const axis = { fill: "var(--muted)", fontSize: 11 };
const grid = "var(--border)";
const colors = ["#356AE6", "#16A6A3", "#D99A26", "#9274E8", "#D95D5D", "#6B7B91"];

export function ImportanceChart() {
  const data = featureImportance.slice(0, 8).map(d => ({ ...d, feature: d.feature.length > 23 ? d.feature.slice(0, 21) + "…" : d.feature }));
  return <ResponsiveContainer width="100%" height="100%"><BarChart data={data} layout="vertical" margin={{top:5,right:20,left:5,bottom:5}}>
    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={grid}/>
    <XAxis type="number" tick={axis} axisLine={false} tickLine={false} />
    <YAxis type="category" dataKey="feature" width={142} tick={axis} axisLine={false} tickLine={false}/>
    <Tooltip contentStyle={{background:"var(--surface)",border:`1px solid var(--border)`,borderRadius:9,color:"var(--text)"}} formatter={(v: number) => [v.toFixed(6),"Importance"]}/>
    <Bar dataKey="value" fill="#356AE6" radius={[0,5,5,0]} barSize={15}/>
  </BarChart></ResponsiveContainer>;
}

export function ImportanceMiniChart() {
  return <ResponsiveContainer width="100%" height="100%"><BarChart data={featureImportance.slice(0,5)} margin={{top:8,right:8,left:-22,bottom:0}}>
    <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false}/>
    <XAxis dataKey="feature" tick={axis} axisLine={false} tickLine={false} interval={0} tickFormatter={(v:string)=>v.length>12?v.slice(0,10)+"…":v}/>
    <YAxis tick={axis} axisLine={false} tickLine={false}/>
    <Tooltip contentStyle={{background:"var(--surface)",border:`1px solid var(--border)`,borderRadius:9,color:"var(--text)"}}/>
    <Bar dataKey="value" fill="#16A6A3" radius={[5,5,0,0]} barSize={26}/>
  </BarChart></ResponsiveContainer>;
}

export function DemoTrendChart() {
  const data = [{x:"Mon",y:0},{x:"Tue",y:0},{x:"Wed",y:0},{x:"Thu",y:0},{x:"Fri",y:0},{x:"Sat",y:0},{x:"Sun",y:0}];
  return <div className="chart-empty-wrap"><div className="chart-empty-icon">↗</div><strong>Connect the prediction API</strong><span>Trend data will appear when real prediction records are available.</span></div>;
}

export function RiskDonut({ flagged, total }: { flagged?: number; total?: number }) {
  if (flagged === undefined || total === undefined || total <= 0) return <div className="chart-empty-wrap"><div className="chart-empty-icon">◌</div><strong>No prediction series yet</strong><span>Run a batch prediction to see the actual risk distribution.</span></div>;
  const data = [{name:"Flagged",value:flagged},{name:"Not flagged",value:Math.max(0,total-flagged)}];
  return <ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={data} dataKey="value" nameKey="name" innerRadius={66} outerRadius={91} paddingAngle={4} stroke="none">{data.map((_,i)=><Cell key={i} fill={i===0?"#D95D5D":"#DCE6F7"}/>)}</Pie><Tooltip/><Legend verticalAlign="bottom" height={28}/></PieChart></ResponsiveContainer>;
}
