import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from io import BytesIO
import json
import base64
import os
from supabase_config import SupabaseAirQualityDB
import pandas as pd

# Page configuration
st.set_page_config(
    page_():
    mainain__"= "__mame__ =

if __nte.analyzer)ession_statats(st.sbase_s_dataay      displ
  cted:er.db_connetate.analyzsion_sd st.sests') anow_db_statate.get('shn_sssio.se  if stn
   Sectiotisticstaase S# Datab
    
              )  v"
    csxt/  mime="te                  v",
}.cs%H%M%S')e('%Y%m%d_ow().strftimtime.ny_data_{dateualit=f"air_qfile_name                  ata,
  csv_d      data=      ,
         CSV"oadDownllabel="📥                 utton(
    ad_bt.downlo       s
                
         dex=False)(in= df.to_csvcsv_data             
    ame(data)d.DataFrf = p           d     
             }
                          ]
            00)
 re'] * 1ticulate_scoresults['pare.sion_stat(st.ses    int                  int'],
  ution_tollresults['psion_state.  st.ses             
         00),* 1y'] sitens['haze_dstate.resulton_st.sessiint(                
        100),_score'] * visibilityesults['ion_state.rsessnt(st.        i              ],
  'category'results[state.ession_        st.s       ,
         'aqi']ts[te.resul.session_sta         st              e': [
 lu'Va                ,
    es (%)']iculatint', 'Parton TPolluti', 'aze (%)'Hity (%)', VisibilCategory', '['AQI', 'Metric':            '        
  = {data         ):
       a"at Dwnload CSVtton("📊 Dobu    if st.        th col3:
 wi      
           )
           n"
   /markdow mime="text                 .md",
  _%H%M%S')}%Y%m%d('.strftime()atetime.nowy_{dlity_summarair_qua=f"e_name        fil            mmary,
data=su              
      ry",ummaad Snlo"📥 Dowbel=    la      
          on(buttst.download_                 
         
      ")}
""tivities']]'acmmendations[ity in recoivy for actactivitn(['- ' + r(10).joines
{chvity Guideli Acti##])}

#ective']ations['protcommend in re measurere for- ' + measu).join(['10chr(easures
{e MtivProtec}

### diate']])mendations['im recommetion infor ac+ action ' ['- n().joi(10ctions
{chrdiate Amme I
###dations
 Recommenealth)}%

## H] * 100_score'teparticularesults['on_state.t.sessiter: {int(s Mataterticul Pagb']}
-y_rlts['skte.resusession_staB{st.Color: RGky - S* 100)}%
sity'] haze_denults['.resssion_state: {int(st.sevelLe
- Haze 100)}%] * _score'bilitylts['visiresusion_state.(st.sesility: {intibisric VpheAtmos
- s Details# Analysi']})

#'categoryesults[state.r{st.session_ (lts['aqi']}te.resuion_stasess*AQI:** {st.M:%S')}
*H:%%Y-%m-%d %ftime('w().strime.noe:** {datet

**Datis Reportty Analys Air Quali"# f""  summary =         ):
     ad Summary"("📝 Downlobutton st.         if  2:
  with col
             
    )            "
  soncation/jpplie="a      mim             .json",
 M%S')}_%H%me('%Y%m%dftime.now().stretiport_{datality_ree=f"air_qu    file_nam               nt=2),
 indeort_data, on.dumps(repta=js          da    
      d JSON","📥 Downloaabel=      l         tton(
     download_bu         st.        
         }
                 s
     mmendation': reconsendatioh_recommealt       'h           esults,
  state.rsion_: st.sess'resultysis_'anal               at(),
     isoformnow().atetime.tamp': d 'times              {
     t_data = or     rep     
      "):N Reportad JSOwnloDo.button("📄        if stl1:
     h co        wit        
s(3)
.columncol3 = stl2, , co     col1
   
        ue)_html=Trnsafe_allow, uv>'dirt Results</xpor">📤 Eeade-hontiec="s<div classn('.markdow  ston
      ctit Sexpor E
        #    
    tml=True)ow_hafe_all""", uns             </div>
           ties']])}
ons['activirecommendatin  item' for i/div>m}<item">{itemendation-ss="recomiv claoin([f'<d.j {''              div>
 Guidelines</y ctivit‍♂️ Aer">🏃adendation-heass="recomm cl       <div
         ">rdon-camendaticomv class="re    <di   "
     ""markdown(ft.           sh col3:
      wit
     e)
      low_html=Trunsafe_al""", u    
             </div>    
   }otective']])dations['prn recommenr item i fodiv>'m">{item}</dation-iteecommeniv class="r([f'<djoin'.   {'       div>
      asures</tive Me">🛡️ Protecaderhen-endatiocomm"rev class=         <di>
       on-card"recommendatidiv class="    <"
        rkdown(f""    st.ma   2:
     with col
               )
 l=Truew_htmlo", unsafe_al      ""
      v>     </di
       te']])}ns['immediamendation recomtem i>' for i}</div>{itemtion-item"da"recommen class=in([f'<div''.jo         {      /div>
 ate Actions< Immedi">🚨der-heaionommendatlass="recv c  <di             ">
 dation-cardcommens="rediv clas    <
        (f"""downst.mark     :
          with col1
     )
        olumns(3t.c2, col3 = s colcol1, 
             'aqi'])
  sults[te.reession_stations(st.srecommendaget_health_lyzer._state.ana st.sessionns =endatioomm        rec     
)
   uetml=Trsafe_allow_h</div>', unndationsh RecommeHealt">🏥 -headersectioniv class="own('<d    st.markd']:
    sss['succestate.resultsion_d st.sesion_state anin st.sesss' f 'resulton
    iecti Stionscommendaealth Re
    # H   image")
 n ocessing ae after prappear herults will is resAnalysst.info("📊                else:
 )
    l=Truew_htmllounsafe_a",        ""
         div>   </             v>
 air</di dust inandes e particlibl">Visetric-desc class="m  <div                  iv>
_pct}%</dticulatelass}">{parlate_c {particuetric-valueass="m   <div cl                </div>
 e Matterat">Particulitle"metric-tass=v cl <di                   💨</div>
-icon">etrics="masiv cl        <d     
       ">metric-card=" <div class             ""
  own(f"markd     st.           ate')
icularte'], 'piculate_scors['partltass(resu_clc_color_metriss = getlarticulate_c  pa             * 100)
  e']te_scorarticulat(results['p inate_pct =  particul              te Matter
ula   # Partic                     
)
        _html=Truefe_allow", unsa     ""    
           </div>          
  iv>:.3f}</dtint']n_ollutiosults['pn Tint: {re">Pollutiometric-desc class="      <div            
  /div>g}, {b})< {{r},B(">RG #667eea;or:yle="collue" stmetric-vaclass="div       <         >
     alysis</divr Anky Coloc-title">S"metriss=iv cla<d                    iv>
🌤️</d">nic-icos="metr clas     <div           ard">
    ="metric-cclassdiv           <   """
   own(f   st.markd             ky_rgb']
results['s, g, b =            r
     nalysisSky Color A #              col_b:
   with              
         rue)
 llow_html=T, unsafe_a      """   >
       </div                /div>
e/smog<ore haz indicates mgecenta>Higher per-desc"ric class="met    <div               v>
 </die_pct}%az">{hlass}ze_cvalue {has="metric-iv clas      <d            </div>
  Levelitle">Haze metric-tdiv class="           <     </div>
    ">🌫️onc-ics="metri   <div clas          ">
       etric-cardss="mladiv c   <          """
   own(f.markd   st             ')
'], 'hazetyensits['haze_dsulclass(reetric_color_ get_maze_class = h            * 100)
   ity'] aze_densts['hresult = int(haze_pc             Level
   # Haze          
                       )
l=Trueallow_htm, unsafe_    """            /div>
          <    /div>
  learer air<tes ccaindie er percentagsc">Highetric-dev class="mdi      <              div>
_pct}%</lityvisibi">{}ssility_cla {visibtric-valueass="mecl<div                     </div>
VisibilityAtmospheric -title">tricass="mev cl       <di            div>
 </-icon">👁️"metricdiv class=         <      
     -card">s="metricv clas       <di        (f"""
 kdownmar       st.        ')
 lityibi'], 'visbility_score'visis[esultss(r_claolorric_c = get_metity_class   visibil        
      100)score'] *ility_'visiblts[esu(r = intty_pctli   visibi          lity
   c Visibi Atmospheri  #   
           with col_a:                 
     lumns(2)
  l_b = st.cool_a, co       ctrics
      me2x2 grid for   # Create  
            
        _html=True)e_allow>', unsafs</divAnalysietailed eader">🔍 D-hection"slass= cdivn('<.markdow       stlysis
     ailed Ana # Det              
 
        w_html=True)nsafe_allo """, u         </div>
            div>
  time']}</lysis_esults['anad at {ris completeysnalaqi-time">A class="div        <
        y']}</div>['categoresultsory">{r"aqi-categ class=      <div        
  i']}</div>['aqltsresu {QI:">Aer-numbqi"aass=     <div cl           ">
s}clas-card {aqi_ class="aqi<div           
 "kdown(f""mar     st.      ss']
 ['clay']]s['categorult[resi_categories.analyzer.aqion_statesess st.i_class =          aqsplay
   AQI Di         #       
   lts
     sureion_state. st.sessts =       resul
     s']:ccesresults['sute.stast.session_n_state and sion st.sess' if 'result
        i    e)
    l=Truw_htmafe_allons/div>', uts<sullysis Reader">📊 Ana-hess="sectiondiv claarkdown('<st.m     :
   col2
    with     ")
nalysis to start aRL enter a Uto, orho pre ae, captun imagad alease uplonfo("👆 Pt.i     s    e:
     els
      _html=True)fe_allownsa)}</div>', ur"Unknown erro", ""errort(ts.ged: {resulis faileAnalysr">❌ -errousat"stass= cln(f'<divmarkdow  st.               e:
             els              html=True)
e_allow_ unsaf...)</div>',d[:8]}s_ilysiD: {anaatabase (Ied to davResults s💾 aved">="status-sssv cladown(f'<di st.mark                         _id:
      analysis   if                              
                
        )                        n_data
    ta, locatioimage_metadandations, s, recomme      result                          atabase(
to_dve_.sanalyzerstate.aon_t.sessid = sanalysis_i                             
                    
                }                    None
   se != 0.0 elitude e if longongitudtude': l   'longi                               e,
   Non != 0.0 elseudede if latittituatitude': la      'l                              ion_name,
 locat  'name':                                ata = {
  cation_d lo                           0):
     0. !=tudengi or lo != 0.0udeor (latitame on_n   if locati                       
  ata = None location_d                        
                                   }
                     }"
   mage.size[1]aded_iloze[0]}x{upage.siuploaded_imsions': f"{    'dimen                      24,
       10//obytes()) aded_image.tn(uploe_kb': le    'siz                      pg',
      apture.ja_ce 'camerlocals() elsn aded_file' ig') if 'uploture.jpra_cap, 'cameile, 'name'ded_fploa': getattr(uame       'n                   {
      adata = ge_met     ima                  ata
     repare metad        # P                       
                         '])
s['aqiultndations(reslth_recommeer.get_heaalyze.ann_statt.sessions = satiorecommend                           
 connected:zer.db_tate.analyon_sst.sessid _db an save_to         if            
   enabled if  database Save to         #                    
                l=True)
   tme_allow_hunsaf', </div>essfully!mpleted succnalysis co">✅ A-successss="status clakdown('<div    st.mar                 ]:
   uccess'f results['s          i            
             ults
     sults = resion_state.re st.sess                  e)
 maguploaded_ialyze_image(analyzer.ane.atession_st.sresults = st                .."):
    ators.ality indicqu air alyzing🔍 An("nerin st.sp     with          e:
 yzld_anal   if shou       ysis
  rm analerfo  # P            
      ue
    = Tre yzanal    should_                
primary"):pe="ity", tyQualir Analyze Atton("🔬    if st.bu      
       e: els           
ue)l=Tre_allow_htmsaf>', un...</divyzing imageuto-anal Ag">🔄nalyzins="status-a'<div classt.markdown(              rue
  lyze = Tould_ana     sh        nalyze:
    if auto_a               
       e = False
 yz should_anal    er
       is trigglysAna   # 
                     age")
loaded Imcaption="Updth=True, er_wi_containd_image, useage(uploade   st.im        ed_image:
 f upload
        i)
        ml=Truefe_allow_ht/div>', unsa Preview<">📸 Imageheaderon-ss="secti clawn('<divarkdo      st.ml1:
  
    with co    1])
s([1,  st.column =, col2col1 area
    nttein con
    # Ma=True)
    e_allow_htmlsaf""", un      
          </div>     v>
    </di           s
    cene sr indoor-ups oe closeextrem    • Avoid           r>
      ions<b condit/strong>ing<ghtng>good litronsure <s E    •           s<br>
     lysiity anailr visibg> focts</stronant objeong>diststr Include <          •          br>
isible sky<with vong> images</strg>outdoor  Use <stron       •            1.4;">
 height: 9rem; line- 0.font-size:3f8; color: #beeiv style="       <d         v>
</dist Resultsfor Bem;">💡 Tips .5ren-bottom: 0bold; margit: nt-weigh90cdf4; fo"color: #div style=         <">
       rem 0;gin: 1ar: 8px; mdiusrarder-em; bong: 1r; paddilid #3182cet: 4px solef border-2a4365;ackground: #style="b   <div   
       """own(rkd     st.ma  ps:
     analysis_tiw_     if shoection
    Tips s      #
  
        ")ge: {str(e)}to load imaed r(f"❌ Failroert.   s                as e:
 eption xcept Exc     e    
       from URL!")ge loaded "✅ Imast.success(                )
    .content)ponse(BytesIO(respen = Image.o_imageploaded     u          0)
     timeout=1s.get(url,  requeste = respons              
     ry:           turl:
     if           L:")
  image UREnter "t(inputext_st. url =            Input":
  URLd == "🌐methoput_lif in       e 
            ")
    fully!red success captuotocess("📸 Ph      st.suc
          ge)(camera_imae.open= Imagge oaded_ima    upl         mage:
   a_i    if camer
        ) picture""Taket(pust.camera_inage = im    camera_
        ent")ronmnvi eutdoorof oture "Take a pickdown(t.mar     s      re":
 era Captu📷 Cam == "ut_methodf inp     eli     
             lly!")
 uccessfu uploaded ss("📸 Imageccessu       st.         aded_file)
en(uploage.opage = Imploaded_im  u            
  d_file:ploade     if u   
             )sults"
   best reimage for outdoor oad an   help="Upl             bmp'],
 ', 'jpeg', ' 'jpg',pe=['png          ty  ,
     file" imagean"Choose             r(
    ile_uploadeile = st.fd_fdeloa          up  Image":
"📁 Upload d == ethoinput_m if 
       ut handlingage inp # Im
             
  image = Noneaded_ uplo
               .6f")
at="%formue=0.0,  valgitude:","Lonput(umber_in st.ntude =gi lon             n:
  loh col_     wit")
       6f"%.mat=lue=0.0, for", vaude:put("Latitr_in= st.numbee  latitud         
      col_lat:th   wi       (2)
   ns= st.columl_lon , co   col_lat        )
 name:"ation input("Loc.text__name = st  location
          tml=True)_allow_h unsafe>',onal)</divtion (Optiocaader">📍 Lction-hesev class="dirkdown('<      st.ma
      ected:r.db_conn.analyzestatession_and st.seto_db save_     if 
   ptional)n input (o  # Locatio  
      )
      alue=True tips", vsisow analykbox("Sh.checs = sttiplysis_ow_ana       sh)
 ectedzer.db_connnalyion_state.at st.sesssabled=nolue=True, di, vae"s to databasve resultkbox("Sa st.chec_db =e_to      save)
  =Tru", valuen uploadalyze o("Auto-ant.checkbox= so_analyze  aut)
       tml=Truefe_allow_h unsas</div>',⚙️ Settingon-header">ectiv class="skdown('<di st.mar     
  gsettin       # S      
 
     )"
     ethod"input_m key=      ,
     ut"]np🌐 URL Ire", "aptu Camera C, "📷mage" I Upload        ["📁,
    hod:"se input methoo      "Cdio(
      hod = st.ra   input_met
            rue)
 low_html=Tnsafe_alut</div>', uImage Inp">📸 ion-header="sectdiv classmarkdown('<       st.       
 
 ction()e_conneup_databasset
        ion setup connectDatabase   # :
     t.sidebar with sSidebar
       
    # )
ruetml=Tallow_h"", unsafe_   "v>
 
    </distorage</p>abase h datquality witr lyze ainao aimages tpture load or ca    <p>Uph1>
    lyzer</ Image Anar Quality <h1>📸 Ai">
       ain-headerclass="m <div    ""
down("rk   st.ma
  Header  
    #ithDB()
  erWAnalyz AirQualitynalyzer =tate.an_ssessio   st.
     state:ion_ssseot in st.'analyzer' ner
    if lyz ana Initialize #:
   def main()e)}")

ics: {str( statistabase dated to loadrror(f"Fail   st.e
     s e:on aeptiExct     excep     
e)
   _width=Trucontainer(fig, use__charttly st.plo                       
 )
                   )
     b=00, t=30,0, r=n=dict(l=  margi        
          ight=300,       he           'white',
  _color=       font             ',
,0,0)0,0a(or='rgb_bgcol       paper          0)',
   ,0,ba(0,0color='rg   plot_bg           
      yout(ate_lafig.upd                   
           )
       )           =8)
ize=dict(s      marker          h=3),
    eea', widt(color='#667line=dict                   
 QI',rage AAveame='   n            ,
     ers'rk+mae='lines   mod             ,
    avg_aqi']['     y=df          '],
     ['date   x=df               atter(
  _trace(go.Scddfig.a           
     Figure() = go.         fig      :
  df.empty if not           
           ly_data)
 rame(dai = pd.DataF    df      own']
  kdbreaaily_ats['d = stdata     daily_   
             =True)
   low_htmlalsafe_v>', un 7 Days)</di Trend (Lastder">📈 AQI-hea"sectionass=clv dikdown('<st.mar    
        eakdown'):('daily_brstats.getif     art
    AQI trend ch    #     
    rue)
    html=Te_allow_v>', unsaf</dirkdown('.mast            
    e)w_html=Truunsafe_allo,    """              
        </div>          span>
     ]}</qi'analysis['ae">AQI {b-stat-valuclass="d  <span                 >
      p}</spantimestam>{label"tat-b-s class="dan  <sp            
          t">db-stass="  <div cla              ""
    arkdown(f"       st.m   
          M')%d %H:%trftime('%m/0')).s'+00:0('Z', eplace'].rated_s['creatalysit(anormaisoftetime.from = datimestamp                  
  s[:3]:nalyserecent_an lysis i    for ana          =True)
  allow_html', unsafe_</div>ysesnt Anal>Rece: 0.5rem;"rgin-bottomld; maight: bont-we8f0; fo2e #eor:olstyle="c('<div kdownar       st.m         ml=True)
llow_ht', unsafe_a-card">ss="db-infoiv clan('<d st.markdow              s:
 yse_analcentre  if      :
     th col3   wi 
       rue)
     w_html=Tfe_allo"", unsa   ">
         iv</d    v>
        </di          an>
      /sp}<dous', 0)get('hazart.gory_disate-value">{catb-stlass="d  <span c         
         pan>ardous</s-label">Hazb-stats="dpan clas       <s       >
      t"="db-sta class     <div     
        </div>              )}</span>
lthy', 0'unheadist.get(tegory_ue">{cab-stat-valass="d clspan  <                  n>
pahealthy</sabel">Unstat-ldb-class="     <span            t">
    ss="db-sta    <div cla       
       </div>      n>
        0)}</spae', oderatet('megory_dist.g">{cat-value"db-statss=clan     <spa         
       ity</span>ate Qualabel">Moderb-stat-lass="dan cl   <sp         
        stat">ass="db-     <div cl        div>
       </            
, 0)}</span>'good'ist.get({category_d">valueat-ass="db-st    <span cl           an>
     ality</spod Qut-label">Gos="db-sta claspan          <s         at">
 ass="db-stdiv cl   <           
  -card">info"db-ass=   <div cl   "
      rkdown(f""       st.ma   n', {})
  ributio_dist'category= stats.get(ist _degory       cat
     col2:with          
       html=True)
nsafe_allow_", u  ""      >
         </diviv>
        </d              
 0)}</span>aqi', ax_et('me">{stats.gdb-stat-valuclass="   <span               
   pan>ax AQI</s-label">Mtat="db-sassn clspa         <       at">
    ass="db-st  <div cl             div>
           </
      span>i', 0)}</('average_aq.get">{statsueval="db-stat-class     <span                >
/spange AQI<">Averatat-labeldb-s="n class     <spa               at">
stlass="db-<div c             
   v>     </di        </span>
   , 0)}yses'total_anals.get('lue">{stat"db-stat-vaspan class=        <         </span>
   s (7 days)sel Analylabel">Totas="db-stat-<span clas             >
       "="db-stat <div class       ">
        b-info-cardv class="ddi      <      """
arkdown(f   st.m  
        with col1:        
  3)
     mns(t.colul3 = sl2, co, co      col1
  
        mit=5)alyses(lint_anceget_rezer.db.analyses = t_analyecen       rys=7)
 (datics.get_statiszer.dbs = analy      stat
  tisticst staGe    # ry:
    
    t)
    uew_html=Trallonsafe_v>', ucs</diase Statisti">📊 Databaderction-heclass="sev rkdown('<di 
    st.maeturn
          rted:
 _connecdbr.yzeot analif n"""
    sticsase statiy datablaDisp"""   lyzer):
 tats(anaabase_ssplay_datef diy")

dand ke URL e bothvid"Please pro st.error(                else:
               ()
    t.rerun         s          key
  = supabase_e_keyupabasn_state.ssio  st.ses             _url
      = supabaserlsupabase_uion_state..sess        st            ey:
_ksupabasel and ase_urpab su        if
        atabase"):t to Dnecbutton("Con     if st.               
   word")
 type="pass Key:", non"Supabase At(t.text_inpubase_key = ssupa           word")
 passype=" URL:", tupabaset("Spu.text_ine_url = st     supabas
       on"):ectibase Conn Data"🔧 Setupr.expander(debath st.si     wi
   orme setup ftabasDa        #   
rue)
      ml=Tafe_allow_htunsdiv>', nected</Not Conse Databaed">❌ b-disconnect class="drkdown('<div.sidebar.ma       ste:
 ls e    
   e
        stats = Trub_e.show_dsession_stat    st.       ats"):
 e Stabas"📊 Show Dattton(ebar.bu  if st.sidtics
      istatase sdatabow       # Sh     
  e)
   =Trulow_htmlfe_alunsadiv>', nected</e Conasab">✅ Dat-connectedass="dbv clarkdown('<di.sidebar.m        stconnected:
db_ analyzer.r andif analyze  alyzer')
  ate.get('ant.session_st = sanalyzertatus
     sconnection current Check# 
      True)
  l=low_htmsafe_al, un</div>'onctie Conneabas>🗄️ Dat"on-header="secti'<div classr.markdown(sideba
    st.""sidebar"ion in ctneon ctabaseda""Setup  "tion():
   se_connecdatabap_
def setuturn ''
igh'
    rete-hn 'particularetur           else:
 '
        te-moderaten 'particula     retur
        <= 0.6: elif value
       ate-low'ulturn 'partic          re:
   0.3if value <=        ':
rticulatetype == 'patric_    elif meigh'
-hreturn 'haze          :
          else'
ateerhaze-modurn '  ret   6:
       0.e <= aluf v eliw'
       -lozeeturn 'ha      r
      = 0.3: if value <       ':
aze'h_type ==  elif metricpoor'
   ility-sib return 'vi        
   se:
        elmoderate'y-ilitn 'visibetur           r
 ue >= 0.4:f val     eli   ood'
ty-gn 'visibiliretur     :
       >= 0.7   if value ity':
      'visibilric_type == if met"
   "e"nd typvalue aetric based on mlor class Get co"""    type):
 metric_alue,r_class(vtric_cologet_mef  None

de     return")
       str(e)}se: {ve to databa to sar(f"Failed st.erro           e:
ception as    except Exs_id
     alysian     return   )
             
    ymous')'anonser_id', 'uet(te.gtasion_sid=st.seser_      us          ta,
location_dan_data=atio        loc       metadata,
 e_adata=imagage_met      im       ions,
   atmmendh_recoealtns=htioecommenda health_r               sults,
alysis_res_results=an     analysi      
     sis_result(b.save_analyf.did = sel   analysis_
              try:      
ne
     urn Noret          f.db:
   or not selectedonnf.db_celot s     if n"""
    to databaseltss resunalysi"Save a""     e):
   ta=Non_daocationNone, lge_metadata=tions, imandah_recomme healtresults,is_alys anase(self,abe_to_datsavdef    
  }
       
        ]       "
         datesty up air qualiboutmed a📱 Stay infor   "            ",
     tical needsy for crivices onlgency serer🏥 Em        "          ,
  or work"utdond oravel asential tll non-es🚗 Avoid a         "          te",
 e or go remohould closoffices sSchools and "🏫                     ': [
ctivities         'a             ],
 "
         ssaryutely neceless absol unid driving"🚗 Avo                  
  sible", access readilycationcy mediemergen   "💊 Have                fiers",
  rir puaimultiple ors, use ows and doind w Seal         "🏠   ",
        leabe unavoidsuror expooutdoif s ratore P100 respi😷 Us        "         
   ctive': [     'prote              ],
            es"
 ry issuve respirato you haer ife providarhealthc"📞 Contact                   
  ing",unnstems rtion syiltraith air fdoors w🏠 Stay in"               ,
     toms"ing sympncf experiention iedical atte Seek m   "🏥          
       re",xposu outdoor e Avoid all EMERGENCY:ALTH "🚨 HE                 [
   ate':     'immedi        {
        return        us
azardo  # Hse: el      }
                  ]
  "
         ivitiesoutdoor actsential e non-esPostpon "🚶‍♀️                  ",
  estiivi actncel outdoor or cahould limit sols   "🏫 Scho                 le",
ch as possibdoors as muren inep child    "👶 Ke                ts",
enrts evd spoxercise antdoor eancel ou🏃‍♂️ C          "          [
 ities':  'activ            ],
             es"
     in vehicln mode ecirculatior rUse ai"🚗             ",
        iesitenuous activd str avoied and hydrat Stay well       "💧            
 s closed",ownd keep windrs apurifie Use air          "🏠      
     utside",oing o when gKN95 masksor 95 "😷 Wear N                   [
 ive':   'protect      ,
               ]    "
     xposurer eoonged outdoloprAvoid     "🔴            ,
     ne"everyoienced by xperay be e effects m⚠️ Health       "             ,
ble groups"nera vulecially espe,ossibldoors when p intay       "🏠 S             vities",
r actiood limit outdshoulne eryo"🚨 Ev              
      iate': ['immed             {
      return       ealthy
   0:  # Unhf aqi <= 20        eli         }
  ]
               eople"
   pmosty safe for generallalks are t w "🚶‍♀️ Shor                 ",
  tiesorts activioor spty of outde intensi  "🏫 Reduc           
       toms",mpatory syany respirldren for  Monitor chi        "👶         ",
   ive groupsn for sensitertiod outdoor exngemit prolo   "🏃‍♂️ Li                
 vities': ['acti             ],
                   s"
ffic areaigh-trae in h Limit tim        "🌬️     
       ions",ditory con respirats handy fore medicationp rescu     "💊 Kee         
      ,indoors"able vailrifiers if ar pu "🏠 Use ai                 ,
   outdoors"ng masksider wearinsals coindividuitive Sens   "😷                  ve': [
protecti         ',
               ]"
        sonditionatory ce respirhavity if you  air qualMonitor    "👀           s",
      tieal activitinue normtion can conpula✅ General po       "            es",
  issu minory experienceuals mavidtive indinsi    "🔍 Se             ople",
    for most peacceptables uality iir q A      "⚠️              
te': [   'immedia    {
         n      retur
       ateder0:  # Moif aqi <= 10
        el         }         ]
      ded"
    commenrey ighl are hnd jogginging aWalk    "🚶‍♀️                ",
  sports andiesor activitol outdo Normal scho  "🏫                ,
  ies" activitoorll outdin age en can engadr👶 Chil       "           
  y safe",eletare compls setdoor exercill ou♂️ A        "🏃‍   
         ities': [ctiv   'a             ],
           eas"
     ational arutdoor recreks and onjoy par"🌳 E                ,
    ivities"r actdooutnd og aliny for cyc da♂️ Great "🚴‍               
    ation",al ventiln for naturows opeep wind    "🪟 Ke            ",
    ts indoorsplanng  air-purifyiider adding"🌱 Cons               
     e': [ectivrot          'p    ],
                de"
   play outsi tofor childrenafe  S      "👶        ",
       sportsse andercir extions fo condierfect    "🏃‍♂️ P                ",
cessary ne precautionshealth    "✅ No              
   ",activitiesl outdoor ent for alis excelllity "🌟 Air qua                    te': [
immedia      '         urn {
      retood
         # Gf aqi <= 50: i   ""
     on AQI"sedbaations endmmhealth recoive prehens com""Get       "lf, aqi):
 ns(sedatiolth_recommenea get_hdef
    '
     'Hazardousurn     retategory
      return c          :
   ['range'][1]= info <= aqi0] <o['range'][      if inf):
      ries.items(f.aqi_categon sel, info iategory     for c
   ):y(self, aqitegorf _get_ca   
    de       }
   tr(e)}'
   : {s failedsisr': f'Analy    'erro        lse, 
    s': Faces'suc          n {
      retur        e:
     tion asxcep    except E 
          }
             )
    M:%S'me('%H:%fti.now().strtimedatesis_time':       'analy      ),
    ex, 3n_indollutioound(p rx':ution_indell         'po  
     t(avg_b)],(avg_g), in), int [int(avg_r':sky_rgb          ' 3),
      re,ulate_scound(partic roore':culate_scti  'par            ,
  _tint, 3)llution': round(pon_tintio    'pollut           ity, 3),
 nd(haze_densy': rou'haze_densit               
 re, 3),sibility_scod(viscore': rounvisibility_         '       ory,
': categorycateg          'i,
      ted_aq: estima      'aqi'         
 e,ccess': Trusu      '          return {
          
          
    mated_aqi)y(estiget_categor = self._   category)
         0))_index * 40llutionin(500, po10, mnt(max(i = imated_aq   esti           
     )
              
   core * 0.25iculate_s        part
         + * 0.20tion_tintpollu        +
        .30 ity * 0ensaze_d   h          
   ) * 0.25 +ility_score visib1 -      (        (
  ex = ollution_ind        p   late AQI
 lcu5. Ca        #     
    0)
        / 800e_variance n(1.0, nois = mire_scoculaterti        pa    n)
ar(laplacianp.vance = rise_va   noi)
         CV_64F2.ray, cvlacian(gv2.Lap claplacian =         on
   etecti Darticulate# 4. P      
                  gb)
/ total_r - avg_b) g_g(avg_r + avt = max(0, _tin pollution         
   1_b +g_g + avg_r + aval_rgb = avg         toton tint
   e polluticulat      # Cal          
        :, 2])
region[:, .mean(sky_b = npvg_      a  , 1])
    [:, :ky_regionnp.mean(s=      avg_g 
       :, 0]), on[:regi(sky_ np.mean_r =vg         a:, :]
   ight//3, mg_array[:hey_region = i      sk2]
      ray.shape[:th = img_arwidheight,             is
nalysolor A. Sky C# 3                
        
uration))at0 - avg_s, min(1, 1.ty = max(0aze_densi    h0
        ion) / 255.turatnp.mean(sation = atura       avg_s
     :, :, 1]v[ion = hs saturat        GB2HSV)
    cv2.COLOR_Ray,arrColor(img_vt= cv2.c  hsv      lysis
     Haze Ana 2.   #
                2)
       .3) *y * 0nsitde 0.7 + edge_ *ntrastmin(1.0, (co = ility_scoreisib v          pe[1])
 es.sha* edg0] shape[ / (edges.> 0)dges = np.sum(esity dge_den         e, 150)
   ay, 50(gr cv2.Canny edges =          
 255.0 / ray)np.std(gst =      contra  
     ty Analysis Visibili        # 1.              
GRAY)
  _BGR2.COLOR cv2r(img_cv,2.cvtColo cv   gray =      BGR)
   COLOR_RGB2rray, cv2.img_a.cvtColor(_cv = cv2       img   at
  CV formenvert to Op   # Con         
      
      t'}e formavalid imag 'Inor':rrFalse, 'e{'success': urn  ret           = 3:
    shape[2] !r img_array.) != 3 o.shapeimg_arrayf len(   i      
   lidation # Basic va                
     
  array(image)y = np._arra    img     rray
   y aage to numprt PIL im   # Conve:
                 try""
efore)"(same as bcators ty indior air qualiyze image f"""Anal        ):
 imageimage(self,def analyze_
    
    d = Falseconnecte  self.db_       }")
   : {str(e)ailedon fectiase conn"❌ Datab  print(f   
       e:ion as t Exceptep      exc       
          ")
 ot foundials nedentbase cr️ Data("⚠rint       p       lse:
          e
    ")ccessfullyected sue connbas✅ Data"nt(         priue
       cted = Trne.db_con self        y)
       pabase_kesu, abase_urlDB(suptyirQualiaseA= Supab.db        self         base_key:
and supase_url    if supaba           
    y')
      abase_ket('supte.gesion_stat.sesY') or sASE_ANON_KE('SUPABenvet_key = os.g   supabase
         se_url')bat('supaon_state.gest.sessiSE_URL') or 'SUPABA= os.getenv(_url se     supaba      te
 tasession st or ronmenfrom envils iaget credentTry to   #           ry:

        ton"""ectiatabase conne d"Initializ ""       
self):abase(ef _init_dat    d  
base()
  ._init_data        selfd = False
.db_connecteselfne
        lf.db = No  se      tion
e conneclize databastiani      # I        
       }
2a'}
   ': '#742aous', 'color-hazard'aqiss': , 500), 'clae': (201ous': {'rang'Hazard         e'},
   #e53e3olor': 'ealthy', 'c 'aqi-unh 'class':00), 21,: (10': {'range'Unhealthy    '      d8936'},
  '#e 'color': derate',molass': 'aqi-'c51, 100), nge': (rae': {'   'Moderat,
         #48bb78'} 'color': ''aqi-good',, 'class': ge': (0, 50)Good': {'ran   '       = {
   goriestelf.aqi_case       :
 self)f __init__(:
    delyzerWithDBualityAna AirQ
classrue)
ml=Tllow_ht_a, unsafe
"""/style>
    }
<ter;en: cign    text-al   ;
  0.5rem 0in:   marg
     us: 8px; border-radi    rem;
   g: 1     paddin  : white;
 or     col
    #805ad5;background:   
     aved {tus-s  
    .sta  }
  
  : center;ext-align      t;
  em 0: 0.5rgin       mar 8px;
 ius:order-rad  b      m;
g: 1re   paddinhite;
     or: w
        cole3e; #e53ound:backgr  or {
      us-errat
    .st 
   er;
    }align: centxt-    te    0;
 0.5rem rgin:     mapx;
   adius: 8  border-rm;
      padding: 1ree;
        or: whit  col;
      38a169round: #ckg    ba   ccess {
 -su    .status
    ;
    }
centerlign:      text-a;
   in: 0.5rem 0        marg;
s: 8pxdiurder-ra  borem;
      padding: 1
        white;    color: 2ce;
    18und: #3  backgro      lyzing {
-anatus  .sta
  es */ssagStatus me   
    /* 
 
    }ld;ight: bo font-wef0;
       2e8r: #e  colo    
  ue {-stat-val 
    .db    }
   m;
ze: 0.9ret-si     fon;
   lor: #a0aec0       cot-label {
   .db-sta
     }
   m: none;
  tto   border-boild {
     last-cht:.db-sta   
    
 ;
    } #4a5568x solidottom: 1p   border-b
      0.5rem 0;  padding:
      center;lign-items:      aen;
   ace-betweent: spstify-cont   juflex;
     y: la disp{
         .db-stat   
  }
    
  ,0,0,0.2);10px rgba(00 2px :   box-shadow   rem 0;
   : 0.5      margin1rem;
     padding: 2px;
     radius: 1er-bord
        d #4a5568; 1px soli    border:3748;
    d: #2dbackgroun {
        -carddb-info*/
    .s ardbase info c
    /* Data   ;
    }
 or: #e2e8f0    colm;
    size: 0.9re       font-ius: 6px;
 -rader      bordem 0;
  5rmargin: 0.m;
        .75re: 0ngdipad        #667eea;
olid  sr-left: 3px   borde;
     : #1a202cbackground        {
on-item recommendati
    
    .;
    }or: #e2e8f0     col;
    boldont-weight:        f: 1rem;
ottommargin-b       m;
 ap: 0.5re       g center;
 ems:   align-it
     lex;lay: f  disp     er {
 ation-headommendrec
    
    .;
    }2)0.,0,gba(0,0px rw: 0 2px 10dobox-sha
        rem; 1.5ing:dd     pa2px;
   radius: 1    border-568;
    id #4a5er: 1px solbord       8;
 ound: #2d374      backgr {
  rdn-candatio  .recomme*/
  ndations recommelth  /* Hea  
   e; }
  or: #e53e3h { colate-higulartic  .p}
  : #ed8936; te { colorulate-modera
    .partic8bb78; }lor: #4 { coate-lowarticul
    .pe3e; }
    or: #e53high { colaze-    .h8936; }
 #edate { color:aze-moder}
    .h;  #48bb78r:ow { coloze-l    .ha   
3e3e; }
 : #e5 colorty-poor {iliisib.v   
 ed8936; }olor: #te { city-modera   .visibilb78; }
 : #48borolty-good { c.visibili */
    etricsng for mlor codi Co    /*
    
 0;
    } margin:
       e: 0.8rem;-sizfont  0;
      or: #a0aec  col   sc {
   c-detri  .me }
    
      0.5rem 0;
margin:  ;
      bold-weight:       fontem;
  2rfont-size: {
        ue .metric-val
    
    
    }rem;om: 0.5gin-bottar m
       00;: 6nt-weight     fo0.9rem;
    font-size: f0;
       e2e8  color: #  
     {metric-title   
    .
    }
 em;bottom: 0.5rrgin-     ma1.2rem;
   nt-size:    fo {
     iconetric-.m  
    
    }
  ,0,0.2);px rgba(0,0 0 2px 10box-shadow:
        n: center;   text-alig5rem;
     ing: 1.        padd;
: 12pxer-radius   bord
     #4a5568;solid border: 1px 
        ;748#2d3ound:       backgrrd {
  etric-ca
    .m/cards *Analysis *     /
    
 0;
    }n:gimar8;
         0.acity:      op
  rem;size: 0.9     font-me {
   qi-ti
    .a    ;
    }
: 0.5rem 0  margin
      ze: 1.5rem;sit-     fon
   egory {  .aqi-cat
  
       }n: 0;
       margid;
  bolweight:     font- 3rem;
    ont-size:  f      er {
qi-numb
    .a  171b); }
  , #63742a2a, #135degadient(: linear-grackground b-hazardous {qi.a    c53030); }
#e53e3e, #35deg, dient(1r-graround: linea{ backghy healti-un  .aq0); }
  b236, #dd6ed89nt(135deg, #gradied: linear-ounackgr bderate {
    .aqi-mo169); }8, #38ag, #48bb7135deient(ar-gradnd: lined { backgrouoo    .aqi-g    

bold;
    }ght:  font-weie;
       : whit     color;
   3)a(0,0,0,0. 20px rgb: 0 4pxowox-shad  b  em 0;
    n: 1rmargi     
    center;t-align:   tex     ius: 15px;
order-rad       bem;
  padding: 2r        {
-card .aqiCard */
   
    /* AQI 
        }.5rem;
   gap: 0     ;
enterign-items: c      al flex;
  ay:pl      disem 0;
  1.5rem 0 1rmargin: 
        t: bold;nt-weigh       fo1.3rem;
 ize:    font-s    #e2e8f0;
    color:     -header {
 ion.sect
    s */on header /* Secti  
   
  m 0;
    }: 0.5re     margin
   ;nline-block: i     displayrem;
   0.9size: ont-      f 20px;
  radius:order- b      rem 1rem;
 0.5 padding:     te;
     color: whi    3e3e;
   #e5nd: backgrou      ed {
 disconnect  .db-}
    
  
    rem 0;gin: 0.5 mar      
 k; inline-blocay: displ        0.9rem;
t-size:   fon0px;
     adius: 2border-r     rem;
   g: 0.5rem 1      paddin
  white;     color: 
   : #38a169;ound backgr   
    nected {db-con .rs */
    indicatobase status/* Data    
    

    }e: 1.1rem;izont-s   f     : 0.9;
acity op;
       .5rem 0 0 0  margin: 0
      header p {n-
    .mai   }
    old;
 t-weight: b fon       
: 2.5rem;sizet-
        fongin: 0;ar
        mheader h1 {   .main-   
 
    }
 , 0.3);23402, 126, gba(1 r 20pxw: 0 4pxox-shado
        b2rem;gin-bottom:        mar center;
 n:lig    text-ae;
    itlor: wh       cous: 15px;
 order-radi        b: 2rem;
    padding
    00%);4ba2 1#76, 7eea 0%, #66ent(135degr-gradi lineakground: bac       -header {

    .maindient */er graadain he 
    /* M }
   e;
   or: whit col      ;
 e1e1e#1d-color:   backgroun    App {
      .stling */
styheme ark t/* D   le>
 "
<stykdown(""
st.maryling)database stth efore but wisame as bSS (tom C

# Cus"
)ndedxpar_state="ebadeinitial_si,
    "wide"ayout=   lon="📸",
 _ic  page  e",
h Databas witnalyzerlity Image Aua"📸 Air Qle=tit