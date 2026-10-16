 "use client";
import { Check, Monitor, Moon, Sun, SlidersHorizontal } from "lucide-react";
import { PageLayout } from "@/components/page-layout";
import { Panel, SectionHeader } from "@/components/ui";
import { useTheme } from "@/components/theme-provider";
const accents=[{id:"blue",label:"Ocean blue",color:"#356AE6"},{id:"teal",label:"Teal",color:"#0F8B8D"},{id:"violet",label:"Violet",color:"#7657D6"},{id:"graphite",label:"Graphite",color:"#536273"}] as const;
export default function SettingsPage(){
 const {theme,setTheme,accent,setAccent,compact,setCompact}=useTheme();
 return <PageLayout title="Settings" subtitle="Tune the interface to your workflow. Preferences are saved in this browser.">
  <div className="grid two-col"><Panel><SectionHeader title="Appearance" description="Choose a comfortable workspace theme."/>
   <div className="preference-row"><div><strong>Color mode</strong><p>Switch between light and dark surfaces.</p></div><div className="segmented"><button className={theme==="light"?"selected":""} onClick={()=>setTheme("light")}><Sun size={13} style={{display:"inline",verticalAlign:"middle",marginRight:5}}/>Light</button><button className={theme==="dark"?"selected":""} onClick={()=>setTheme("dark")}><Moon size={13} style={{display:"inline",verticalAlign:"middle",marginRight:5}}/>Dark</button></div></div>
   <div className="preference-row"><div><strong>Accent color</strong><p>Applied to navigation, buttons, and visual emphasis.</p></div><div className="color-options">{accents.map(a=><button key={a.id} title={a.label} aria-label={a.label} onClick={()=>setAccent(a.id)} className={`color-swatch ${accent===a.id?"selected":""}`} style={{background:a.color}}>{accent===a.id&&<Check size={13} color="#fff"/>}</button>)}</div></div>
   <div className="preference-row"><div><strong>Data density</strong><p>Use compact rows for information-heavy tables.</p></div><div className="segmented"><button className={!compact?"selected":""} onClick={()=>setCompact(false)}>Comfortable</button><button className={compact?"selected":""} onClick={()=>setCompact(true)}>Compact</button></div></div>
  </Panel><Panel><SectionHeader title="Interface preview" description="Changes apply immediately."/>
   <div className="settings-preview"><div className="preview-top"><span className="preview-mark"><Monitor size={15}/></span><div><strong>Shipment risk overview</strong><small>Preview card</small></div><span className="preview-pill">Active</span></div><div className="preview-value">71.99%</div><div className="preview-sub">Recorded Phase 5 test F1 score</div><div className="preview-track"><span/></div><div className="preview-bottom"><span>Model performance</span><strong>Held-out test</strong></div></div>
   <div className="alert section-gap"><SlidersHorizontal size={16}/><div>Preferences are stored locally in your browser. They do not change the ML model, predictions, or evaluation metrics.</div></div>
  </Panel></div>
 </PageLayout>;
}
