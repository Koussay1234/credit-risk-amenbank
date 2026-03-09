import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Amen Bank — Analyse Risque Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

USERS = {
    "admin":      {"password":"amen2026",   "role":"Directeur Risque",  "nom":"Ahmed Ben Salah",  "avatar":"👨‍💼"},
    "analyste1":  {"password":"analyse123", "role":"Analyste Crédit",   "nom":"Sarra Maaloul",    "avatar":"👩‍💼"},
    "koussay":    {"password":"pfe2026",    "role":"Stagiaire PFE",     "nom":"Koussay Hassana",  "avatar":"👨‍🎓"},
}

AMEN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&family=Open+Sans:wght@300;400;500;600&display=swap');
:root {
    --navy:#002B5C; --blue:#004A99; --red:#C8102E;
    --gold:#B8941F; --bg:#F4F7FB; --white:#FFFFFF;
    --gray:#6B7280; --border:#D1DCE8; --success:#1A7A4A;
}
*, *::before, *::after { box-sizing:border-box; }
html,body,.stApp { background:var(--bg)!important; font-family:'Open Sans',sans-serif; color:#1A2B45; }
.block-container { padding:1.5rem 2rem!important; max-width:1400px!important; }
[data-testid="stSidebar"] { background:var(--navy)!important; border-right:3px solid var(--red)!important; }
[data-testid="stSidebar"] .block-container { padding:0!important; }
[data-testid="stSidebar"] * { color:white!important; }
.stButton>button { font-family:'Montserrat',sans-serif!important; font-weight:600!important; border-radius:6px!important; border:none!important; transition:all 0.2s!important; }
.stSelectbox>div>div { border-radius:6px!important; border:1px solid var(--border)!important; }
.stNumberInput>div>div>input { border-radius:6px!important; border:1px solid var(--border)!important; }
.stProgress>div>div { background:linear-gradient(90deg,var(--blue),var(--red))!important; border-radius:100px!important; }
.stProgress>div { background:#D1DCE8!important; border-radius:100px!important; height:10px!important; }
.stDataFrame { border-radius:8px!important; overflow:hidden!important; }
#MainMenu,footer,header { visibility:hidden; }
</style>
"""

def amen_header(title, subtitle="", icon=""):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#002B5C,#004A99);padding:1.5rem 2rem;border-radius:10px;margin-bottom:1.5rem;border-left:5px solid #C8102E;display:flex;align-items:center;gap:1rem;">
        <div style="font-size:2rem;">{icon}</div>
        <div>
            <div style="font-family:'Montserrat',sans-serif;font-size:1.3rem;font-weight:800;color:white;">{title}</div>
            <div style="font-size:0.78rem;color:#A8C4E0;margin-top:2px;">{subtitle}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def kpi_card(label, value, color="#004A99", badge="", badge_color=""):
    badge_html = f"<div style='margin-top:6px;font-size:0.68rem;padding:2px 8px;border-radius:20px;display:inline-block;background:{badge_color};color:white;font-weight:600;'>{badge}</div>" if badge else ""
    st.markdown(f"""
    <div style="background:white;border:1px solid #D1DCE8;border-radius:10px;padding:1.2rem;text-align:center;border-top:3px solid {color};box-shadow:0 2px 8px rgba(0,43,92,0.08);">
        <div style="font-family:'Montserrat',sans-serif;font-size:1.7rem;font-weight:800;color:{color};">{value}</div>
        <div style="font-size:0.72rem;color:#6B7280;margin-top:4px;font-weight:500;">{label}</div>
        {badge_html}
    </div>""", unsafe_allow_html=True)

def section_title(text):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 0.8rem;">
        <div style="width:4px;height:20px;background:#C8102E;border-radius:2px;"></div>
        <span style="font-family:'Montserrat',sans-serif;font-size:0.85rem;font-weight:700;color:#002B5C;text-transform:uppercase;letter-spacing:1px;">{text}</span>
    </div>""", unsafe_allow_html=True)

def divider():
    st.markdown('<div style="height:1px;background:#D1DCE8;margin:1.2rem 0;"></div>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        xgb = joblib.load("meilleur_modele_xgb.pkl")
        nn = joblib.load("nn_wrapper.pkl")
        sc = joblib.load("scaler.pkl")
        return xgb, nn, sc, True
    except:
        return None, None, None, False

@st.cache_data
def load_data():
    try:
        return pd.read_csv("credit_customers_PFE_francais.csv", sep=";", encoding="latin-1")
    except:
        return None

if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
if "username"    not in st.session_state: st.session_state.username    = ""
if "historique"  not in st.session_state: st.session_state.historique  = []
if "login_error" not in st.session_state: st.session_state.login_error = ""

# ══════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown(AMEN_CSS, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .stApp { background:linear-gradient(135deg,#002B5C 0%,#004A99 60%,#001F45 100%)!important; }
    .block-container { max-width:480px!important; margin:auto!important; padding-top:4rem!important; }
    </style>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem;">
        <div style="font-size:4rem;">🏦</div>
        <div style="font-family:'Montserrat',sans-serif;font-size:2.2rem;font-weight:800;color:white;letter-spacing:4px;">AMEN BANK</div>
        <div style="font-size:0.72rem;color:#C8102E;letter-spacing:3px;text-transform:uppercase;font-weight:700;margin-top:6px;background:rgba(255,255,255,0.1);display:inline-block;padding:4px 16px;border-radius:20px;">
            Système d'Analyse du Risque Crédit
        </div>
    </div>
    <div style="background:white;border-radius:14px;padding:2.5rem;box-shadow:0 20px 60px rgba(0,0,0,0.3);border-top:5px solid #C8102E;">
        <div style="text-align:center;margin-bottom:1.5rem;">
            <div style="font-family:'Montserrat',sans-serif;font-size:0.9rem;font-weight:700;color:#002B5C;text-transform:uppercase;letter-spacing:2px;">Connexion Sécurisée</div>
            <div style="height:2px;background:linear-gradient(90deg,transparent,#D1DCE8,transparent);margin-top:10px;"></div>
        </div>
    """, unsafe_allow_html=True)

    username = st.text_input("👤  Identifiant", placeholder="Entrez votre identifiant")
    password = st.text_input("🔒  Mot de passe", type="password", placeholder="Entrez votre mot de passe")

    if st.session_state.login_error:
        st.markdown(f"""<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:0.8rem;text-align:center;margin:0.5rem 0;">
            <span style="color:#C8102E;font-size:0.82rem;font-weight:600;">❌ {st.session_state.login_error}</span></div>""",
            unsafe_allow_html=True)

    login_btn = st.button("🔐  SE CONNECTER", use_container_width=True, type="primary")

    if login_btn:
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in   = True
            st.session_state.username    = username
            st.session_state.login_error = ""
            st.rerun()
        else:
            st.session_state.login_error = "Identifiant ou mot de passe incorrect."
            st.rerun()

    st.markdown("""
    <div style="height:1px;background:#D1DCE8;margin:1.5rem 0;"></div>
    <div style="text-align:center;font-size:0.7rem;color:#9CA3AF;line-height:1.8;">
        🔐 Accès réservé au personnel autorisé<br>
        Amen Bank · Direction des Risques · 2025-2026<br>
        <span style="color:#C8102E;font-weight:700;">PFE — Koussay Hassana</span>
    </div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:1rem;margin-top:1rem;text-align:center;">
        <div style="font-size:0.7rem;color:#A8C4E0;line-height:1.8;">
            🔐 Accès strictement réservé au personnel habilité<br>
            Pour obtenir vos identifiants, contactez votre responsable hiérarchique.<br>
            <span style="color:#C8102E;font-weight:700;font-size:0.68rem;">Toute tentative d'accès non autorisé est enregistrée.</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# APP PRINCIPALE
# ══════════════════════════════════════════════════════════════
else:
    st.markdown(AMEN_CSS, unsafe_allow_html=True)
    user_info = USERS[st.session_state.username]
    xgb_model, nn_model, scaler, models_ok = load_models()
    df_original = load_data()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f"""
        <div style="background:linear-gradient(180deg,#001F45,#002B5C);padding:1.5rem;border-bottom:3px solid #C8102E;">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="font-size:2rem;">🏦</div>
                <div>
                    <div style="font-family:'Montserrat',sans-serif;font-size:1.3rem;font-weight:800;color:white;letter-spacing:2px;">AMEN BANK</div>
                    <div style="font-size:0.6rem;color:#C8102E;letter-spacing:2px;font-weight:700;text-transform:uppercase;">Risque Crédit IA</div>
                </div>
            </div>
        </div>
        <div style="padding:1rem 1.2rem;background:rgba(255,255,255,0.06);border-bottom:1px solid rgba(255,255,255,0.1);">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="font-size:1.8rem;">{user_info['avatar']}</div>
                <div>
                    <div style="font-weight:700;font-size:0.85rem;color:white;">{user_info['nom']}</div>
                    <div style="font-size:0.68rem;color:#A8C4E0;">{user_info['role']}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="padding:0.8rem 1.2rem 0.3rem;font-size:0.6rem;color:#A8C4E0;letter-spacing:2px;text-transform:uppercase;font-weight:700;">Navigation</div>', unsafe_allow_html=True)

        page = st.radio("", [
            "🔍  Analyse Client",
            "📊  Tableau de Bord",
            "📋  Historique",
            "🤖  Modèles IA",
            "ℹ️  À propos"
        ], label_visibility="collapsed")

        st.markdown('<div style="height:1px;background:rgba(255,255,255,0.1);margin:1rem 1.2rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="padding:0 1.2rem 0.3rem;font-size:0.6rem;color:#A8C4E0;letter-spacing:2px;text-transform:uppercase;font-weight:700;">Statut Système</div>', unsafe_allow_html=True)

        dot_color = "#1FAF70" if models_ok else "#C8102E"
        status_items = [("XGBoost","#1FAF70"),("Réseau Neurones","#1FAF70"),("SHAP","#1FAF70")] if models_ok else [("Modèles non chargés","#C8102E")]
        for label, color in status_items:
            st.markdown(f"""<div style="padding:0 1.2rem;display:flex;align-items:center;gap:8px;padding:4px 1.2rem;">
                <div style="width:7px;height:7px;background:{color};border-radius:50%;box-shadow:0 0 5px {color};"></div>
                <span style="font-size:0.72rem;color:#A8C4E0;">{label}</span></div>""", unsafe_allow_html=True)

        nb_pred = len(st.session_state.historique)
        st.markdown(f"""
        <div style="height:1px;background:rgba(255,255,255,0.1);margin:1rem 1.2rem;"></div>
        <div style="margin:0 1.2rem;background:rgba(200,16,46,0.15);border:1px solid rgba(200,16,46,0.3);border-radius:8px;padding:0.8rem;text-align:center;">
            <div style="font-family:'Montserrat',sans-serif;font-size:1.5rem;font-weight:800;color:#FF6B80;">{nb_pred}</div>
            <div style="font-size:0.65rem;color:#A8C4E0;">Analyses effectuées</div>
        </div>
        <div style="height:1px;background:rgba(255,255,255,0.1);margin:1rem 1.2rem;"></div>
        <div style="padding:0 1.2rem;">""", unsafe_allow_html=True)

        if st.button("🚪  Se déconnecter", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.historique = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── PAGE ANALYSE ──
    if "Analyse" in page:
        amen_header("Analyse du Risque Client","Évaluation IA de la probabilité de défaut de paiement","🔍")

        with st.form("form_client"):
            col1, col2, col3 = st.columns(3, gap="large")
            with col1:
                section_title("Profil Personnel")
                age = st.slider("Âge", 18, 75, 35)
                situation_personnelle = st.selectbox("Situation familiale", [0,1,2,3,4],
                    format_func=lambda x: ["Homme célibataire","Femme div/dép/mariée","Homme div/séparé","Homme marié/veuf","Femme célibataire"][x])
                nombre_personnes_a_charge = st.selectbox("Personnes à charge", [1, 2])
                possede_telephone = st.selectbox("Téléphone enregistré", [0,1], format_func=lambda x: ["Non","Oui"][x])
                travailleur_etranger = st.selectbox("Statut résidence", [0,1], format_func=lambda x: ["Résident tunisien","Travailleur étranger"][x])

            with col2:
                section_title("Profil Financier")
                statut_compte_courant = st.selectbox("Statut compte courant", [0,1,2,3],
                    format_func=lambda x: ["Débiteur (< 0 DT)","Faible (0-200 DT)","Suffisant (> 200 DT)","Aucun compte"][x])
                etat_epargne = st.selectbox("État de l'épargne", [0,1,2,3,4],
                    format_func=lambda x: ["Très faible (< 100 DT)","Faible (100-500 DT)","Moyen (500-1000 DT)","Élevé (> 1000 DT)","Non communiqué"][x])
                montant_credit = st.number_input("Montant demandé (DT)", min_value=250, max_value=200000, value=5000, step=500)
                duree_mois = st.slider("Durée remboursement (mois)", 4, 72, 24)
                taux_remboursement = st.selectbox("Taux effort", [1,2,3,4],
                    format_func=lambda x: f"{[25,50,75,100][x-1]}% du revenu disponible")

            with col3:
                section_title("Informations Crédit")
                credit_history = st.selectbox("Historique bancaire", [0,1,2,3,4],
                    format_func=lambda x: ["⚠️ Historique critique","⚠️ Retards passés","✅ Crédits remboursés","✅ Tous remboursés","➖ Sans historique"][x])
                objectif_credit = st.selectbox("Objet du financement", list(range(10)),
                    format_func=lambda x: ["🚗 Véhicule neuf","🚙 Véhicule occasion","🛋️ Mobilier","📺 Électronique","🏠 Électroménager","🔧 Travaux","📚 Études","✈️ Tourisme","🔄 Refinancement","📦 Autres"][x])
                anciennete_emploi = st.selectbox("Ancienneté emploi", [0,1,2,3,4],
                    format_func=lambda x: ["Sans emploi","< 1 an","1 à 4 ans","4 à 7 ans","> 7 ans"][x])
                categorie_emploi = st.selectbox("Catégorie professionnelle", [0,1,2,3],
                    format_func=lambda x: ["Sans emploi","Non qualifié","Qualifié/Technicien","Cadre/Expert"][x])
                type_logement = st.selectbox("Situation logement", [0,1,2],
                    format_func=lambda x: ["Logé gratuitement","Locataire","Propriétaire"][x])

            divider()
            _, btn_col, _ = st.columns([1,2,1])
            with btn_col:
                submitted = st.form_submit_button("🔍  LANCER L'ANALYSE DE RISQUE", use_container_width=True, type="primary")

        if submitted:
            if not models_ok:
                st.error("❌ Les modèles IA ne sont pas disponibles.")
            else:
                input_data = pd.DataFrame([{
                    'statut_compte_courant':statut_compte_courant,'duree_mois':duree_mois,
                    'credit_history':credit_history,'objectif_credit':objectif_credit,
                    'montant_credit':montant_credit,'etat_epargne':etat_epargne,
                    'anciennete_emploi':anciennete_emploi,'taux_remboursement':taux_remboursement,
                    'situation_personnelle':situation_personnelle,'autres_garants':0,
                    'anciennete_logement':3,'importance_biens':2,'age':age,
                    'autres_plans_paiement':0,'type_logement':type_logement,
                    'nombre_credits_existants':1,'categorie_emploi':categorie_emploi,
                    'nombre_personnes_a_charge':nombre_personnes_a_charge,
                    'possede_telephone':possede_telephone,'travailleur_etranger':travailleur_etranger
                }])

                with st.spinner("Analyse IA en cours..."):
                    proba_xgb  = xgb_model.predict_proba(input_data)[0][1]
                    pred_xgb   = int(proba_xgb >= 0.5)
                    proba_nn   = nn_model.predict_proba(input_data)[0][1]
                    pred_nn    = int(proba_nn >= 0.5)
                    proba_moy  = (proba_xgb + proba_nn) / 2
                    pred_final = int(proba_moy >= 0.5)
                    niveau     = "ÉLEVÉ" if proba_moy>0.6 else "MODÉRÉ" if proba_moy>0.4 else "FAIBLE"
                    c_niv      = "#C8102E" if proba_moy>0.6 else "#D97706" if proba_moy>0.4 else "#1A7A4A"
                    rec        = "REFUSÉ" if pred_final else "APPROUVÉ"
                    c_rec      = "#C8102E" if pred_final else "#1A7A4A"
                    bg_rec     = "#FEF2F2" if pred_final else "#F0FDF4"
                    br_rec     = "#FECACA" if pred_final else "#BBF7D0"
                    icon_rec   = "❌" if pred_final else "✅"

                divider()
                section_title("Résultats de l'Analyse IA")

                st.markdown(f"""
                <div style="background:{bg_rec};border:2px solid {br_rec};border-radius:12px;padding:2rem;text-align:center;margin-bottom:1.5rem;border-left:6px solid {c_rec};">
                    <div style="font-size:2.5rem;">{icon_rec}</div>
                    <div style="font-family:'Montserrat',sans-serif;font-size:1.8rem;font-weight:800;color:{c_rec};">DOSSIER {rec}</div>
                    <div style="font-size:0.85rem;color:#6B7280;margin-top:0.5rem;">
                        Probabilité de défaut : <strong style="color:{c_rec};">{proba_moy*100:.1f}%</strong> — Risque : <strong style="color:{c_niv};">{niveau}</strong>
                    </div>
                </div>""", unsafe_allow_html=True)

                k1,k2,k3,k4 = st.columns(4)
                with k1: kpi_card("XGBoost", f"{proba_xgb*100:.1f}%", "#C8102E" if pred_xgb else "#1A7A4A", "Mauvais" if pred_xgb else "Bon", "#C8102E" if pred_xgb else "#1A7A4A")
                with k2: kpi_card("Réseau Neurones", f"{proba_nn*100:.1f}%", "#C8102E" if pred_nn else "#1A7A4A", "Mauvais" if pred_nn else "Bon", "#C8102E" if pred_nn else "#1A7A4A")
                with k3: kpi_card("Prob. défaut", f"{proba_moy*100:.1f}%", c_rec, "Score combiné", c_rec)
                with k4: kpi_card("Niveau risque", niveau, c_niv, "Évaluation IA", c_niv)

                st.markdown(f"""
                <div style="background:white;border:1px solid #D1DCE8;border-radius:10px;padding:1.2rem;margin:1rem 0;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                        <span style="font-size:0.75rem;color:#6B7280;">Risque faible</span>
                        <span style="font-family:'Montserrat';font-weight:700;color:{c_niv};">{proba_moy*100:.1f}% — {niveau}</span>
                        <span style="font-size:0.75rem;color:#6B7280;">Risque élevé</span>
                    </div>
                    <div style="background:#E5E7EB;border-radius:100px;height:14px;overflow:hidden;">
                        <div style="width:{proba_moy*100:.1f}%;height:100%;background:linear-gradient(90deg,#1A7A4A,#D97706,#C8102E);border-radius:100px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

                section_title("Interprétabilité SHAP — Facteurs de décision")
                try:
                    explainer = shap.Explainer(xgb_model, input_data)
                    shap_values = explainer(input_data)
                    fig, ax = plt.subplots(figsize=(10,4))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                except:
                    imp = xgb_model.feature_importances_
                    top_idx = np.argsort(imp)[-8:][::-1]
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.barh([list(input_data.columns)[i] for i in top_idx], [imp[i] for i in top_idx],
                            color=['#004A99' if imp[i]>np.median(imp) else '#A8C4E0' for i in top_idx], height=0.6)
                    ax.set_xlabel("Importance"); ax.set_title("Importance des variables", fontweight='bold', color='#002B5C')
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

                if pred_final == 0:
                    st.success(f"✅ **Crédit recommandé** — Probabilité de défaut : {proba_moy*100:.1f}%. Profil client favorable.")
                else:
                    st.error(f"❌ **Crédit déconseillé** — Risque de défaut élevé ({proba_moy*100:.1f}%). Demandez des garanties supplémentaires.")

                st.session_state.historique.append({
                    "N°":len(st.session_state.historique)+1,
                    "Analyste":user_info['nom'],"Âge":age,
                    "Montant (DT)":montant_credit,"Durée (mois)":duree_mois,
                    "XGBoost":f"{proba_xgb*100:.1f}%","Réseau Neurones":f"{proba_nn*100:.1f}%",
                    "Prob. défaut":f"{proba_moy*100:.1f}%","Niveau":niveau,
                    "Décision":f"{icon_rec} {rec}"
                })
                st.info(f"📋 Analyse N°{len(st.session_state.historique)} enregistrée.")

    # ── PAGE TABLEAU DE BORD ──
    elif "Tableau" in page:
        amen_header("Tableau de Bord","Vue d'ensemble du portefeuille crédit","📊")
        if df_original is None:
            st.error("Dataset non trouvé.")
        else:
            df = df_original.copy()
            total=len(df); bons=(df['classe']=='bon').sum(); mauvais=(df['classe']=='mauvais').sum()
            moy_m=df['montant_credit'].mean(); moy_a=df['age'].mean(); moy_d=df['duree_mois'].mean()

            k1,k2,k3,k4,k5,k6 = st.columns(6)
            with k1: kpi_card("Total dossiers",str(total),"#002B5C")
            with k2: kpi_card("Bons clients",str(bons),"#1A7A4A",f"{bons/total*100:.0f}%","#1A7A4A")
            with k3: kpi_card("Mauvais clients",str(mauvais),"#C8102E",f"{mauvais/total*100:.0f}%","#C8102E")
            with k4: kpi_card("Montant moyen",f"{moy_m:.0f} DT","#004A99")
            with k5: kpi_card("Âge moyen",f"{moy_a:.0f} ans","#B8941F")
            with k6: kpi_card("Durée moyenne",f"{moy_d:.0f} mois","#6B21A8")

            divider()
            NAVY='#002B5C'; BLUE='#004A99'; RED='#C8102E'; GRAY='#6B7280'

            col1,col2 = st.columns(2, gap="large")
            with col1:
                section_title("Répartition du Portefeuille")
                fig,ax = plt.subplots(figsize=(6,4))
                wedges,texts,autotexts = ax.pie([bons,mauvais],labels=['Bon risque','Mauvais risque'],
                    colors=[BLUE,RED],autopct='%1.1f%%',startangle=90,
                    wedgeprops=dict(width=0.55,edgecolor='white',linewidth=3))
                for t in texts: t.set_color(GRAY); t.set_fontsize(9)
                for at in autotexts: at.set_color('white'); at.set_fontweight('bold'); at.set_fontsize(10)
                ax.set_title("Répartition Bon / Mauvais risque",color=NAVY,fontsize=11,fontweight='bold')
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

            with col2:
                section_title("Distribution par Âge")
                fig,ax = plt.subplots(figsize=(6,4))
                df[df['classe']=='bon']['age'].hist(ax=ax,bins=20,alpha=0.8,color=BLUE,label='Bon',edgecolor='white')
                df[df['classe']=='mauvais']['age'].hist(ax=ax,bins=20,alpha=0.8,color=RED,label='Mauvais',edgecolor='white')
                ax.set_xlabel("Âge",color=GRAY,fontsize=9); ax.set_ylabel("Clients",color=GRAY,fontsize=9)
                ax.tick_params(colors=GRAY,labelsize=8); ax.legend(fontsize=8)
                for spine in ax.spines.values(): spine.set_color('#E5E7EB')
                ax.set_title("Distribution de l'âge",color=NAVY,fontsize=11,fontweight='bold')
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

            col3,col4 = st.columns(2, gap="large")
            with col3:
                section_title("Montant par Classe")
                fig,ax = plt.subplots(figsize=(6,4))
                bp = ax.boxplot([df[df['classe']=='bon']['montant_credit'].values, df[df['classe']=='mauvais']['montant_credit'].values],
                    labels=['Bon','Mauvais'],patch_artist=True,
                    medianprops=dict(color=NAVY,linewidth=2),
                    whiskerprops=dict(color=GRAY),capprops=dict(color=GRAY),
                    flierprops=dict(markerfacecolor=GRAY,markersize=3))
                bp['boxes'][0].set_facecolor(BLUE+'33'); bp['boxes'][0].set_edgecolor(BLUE)
                bp['boxes'][1].set_facecolor(RED+'33');  bp['boxes'][1].set_edgecolor(RED)
                ax.set_ylabel("Montant (DT)",color=GRAY,fontsize=9); ax.tick_params(colors=GRAY,labelsize=8)
                for spine in ax.spines.values(): spine.set_color('#E5E7EB')
                ax.set_title("Montant crédit par classe",color=NAVY,fontsize=11,fontweight='bold')
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

            with col4:
                section_title("Top 5 Objectifs de Crédit")
                fig,ax = plt.subplots(figsize=(6,4))
                obj_counts = df['objectif_credit'].value_counts().head(5)
                colors_bar = [BLUE,NAVY,RED,'#B8941F','#6B21A8']
                bars = ax.barh(range(len(obj_counts)),obj_counts.values,color=colors_bar[:len(obj_counts)],edgecolor='white',height=0.6)
                ax.set_yticks(range(len(obj_counts))); ax.set_yticklabels([str(l)[:15] for l in obj_counts.index],fontsize=8,color=GRAY)
                ax.set_xlabel("Nombre de clients",color=GRAY,fontsize=9); ax.tick_params(colors=GRAY,labelsize=8)
                for spine in ax.spines.values(): spine.set_color('#E5E7EB')
                for bar,val in zip(bars,obj_counts.values):
                    ax.text(bar.get_width()+3,bar.get_y()+bar.get_height()/2,str(val),va='center',fontsize=8,color=NAVY,fontweight='bold')
                ax.set_title("Objectifs de financement",color=NAVY,fontsize=11,fontweight='bold')
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    # ── PAGE HISTORIQUE ──
    elif "Historique" in page:
        amen_header("Historique des Analyses","Traçabilité complète des décisions de crédit","📋")
        if len(st.session_state.historique) == 0:
            st.markdown("""<div style="text-align:center;padding:4rem;background:white;border:2px dashed #D1DCE8;border-radius:12px;">
                <div style="font-size:3rem;">📋</div>
                <div style="font-family:'Montserrat',sans-serif;font-size:1rem;font-weight:700;color:#002B5C;">Aucune analyse enregistrée</div>
                <div style="font-size:0.8rem;color:#6B7280;margin-top:0.5rem;">Effectuez une première analyse pour voir l'historique</div>
            </div>""", unsafe_allow_html=True)
        else:
            df_hist = pd.DataFrame(st.session_state.historique)
            nb_mauvais = df_hist['Décision'].str.contains("REFUSÉ").sum()
            nb_bons = len(df_hist) - nb_mauvais
            k1,k2,k3,k4 = st.columns(4)
            with k1: kpi_card("Total analyses",str(len(df_hist)),"#002B5C")
            with k2: kpi_card("Approuvés",str(nb_bons),"#1A7A4A",f"{nb_bons/len(df_hist)*100:.0f}%","#1A7A4A")
            with k3: kpi_card("Refusés",str(nb_mauvais),"#C8102E",f"{nb_mauvais/len(df_hist)*100:.0f}%","#C8102E")
            with k4: kpi_card("Taux de rejet",f"{nb_mauvais/len(df_hist)*100:.0f}%","#B8941F")
            divider()
            st.dataframe(df_hist,use_container_width=True,hide_index=True,height=400)
            c1,_,c3 = st.columns(3)
            with c1:
                csv = df_hist.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️  Exporter CSV",csv,"historique_amen_bank.csv","text/csv",use_container_width=True)
            with c3:
                if st.button("🗑️  Effacer l'historique",use_container_width=True):
                    st.session_state.historique = []; st.rerun()

    # ── PAGE MODÈLES ──
    elif "Modèles" in page:
        amen_header("Modèles d'Intelligence Artificielle","Performance et comparaison des algorithmes de scoring crédit","🤖")
        perf = pd.DataFrame({
            "Modèle":["XGBoost ⭐ (déployé)","Random Forest","Régression Logistique","Arbre de Décision","Réseau Neurones 🧠 (déployé)"],
            "Accuracy":["74.0%","74.5%","68.5%","60.5%","~74%"],
            "AUC-ROC":["0.7910","0.7733","0.7688","0.5750","~0.79"],
            "Precision":["56.3%","58.2%","47.9%","38.0%","~57%"],
            "Recall":["60.0%","53.3%","58.3%","50.0%","~60%"],
            "F1-Score":["58.1%","55.7%","52.6%","43.2%","~58%"],
        })
        st.dataframe(perf,use_container_width=True,hide_index=True)
        divider()
        col1,col2 = st.columns(2, gap="large")
        with col1:
            section_title("Comparaison AUC-ROC")
            fig,ax = plt.subplots(figsize=(6,4))
            noms=["XGBoost","Random\nForest","Rég.\nLogistique","Arbre\nDécision"]
            aucs=[0.7910,0.7733,0.7688,0.5750]
            bars=ax.bar(noms,aucs,color=['#002B5C','#004A99','#A8C4E0','#D1DCE8'],edgecolor='white',linewidth=2,width=0.5)
            ax.set_ylim(0.4,0.85); ax.set_ylabel("AUC-ROC",color='#6B7280',fontsize=9)
            ax.tick_params(colors='#6B7280',labelsize=8)
            for spine in ax.spines.values(): spine.set_color('#E5E7EB')
            ax.axhline(y=0.5,color='#C8102E',linestyle='--',linewidth=1.5,label='Référence')
            for bar,val in zip(bars,aucs):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{val:.4f}',ha='center',va='bottom',color='#002B5C',fontsize=8,fontweight='bold')
            ax.legend(fontsize=8); ax.set_title("Performance par modèle",color='#002B5C',fontsize=11,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            section_title("Paramètres XGBoost Optimisé")
            for k,v in {"learning_rate":"0.05","max_depth":"7","n_estimators":"300","subsample":"0.7","colsample_bytree":"0.7","Méthode":"GridSearchCV 5-fold","Métrique":"AUC-ROC","Rééchantillonnage":"SMOTE (560/560)"}.items():
                st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:0.6rem 1rem;margin:4px 0;background:white;border-radius:8px;border:1px solid #D1DCE8;">
                    <span style="color:#6B7280;font-size:0.8rem;font-weight:500;">{k}</span>
                    <span style="color:#004A99;font-size:0.8rem;font-weight:700;font-family:'Montserrat',sans-serif;">{v}</span>
                </div>""", unsafe_allow_html=True)

    # ── PAGE À PROPOS ──
    elif "propos" in page:
        amen_header("À propos du Projet PFE","Documentation et informations sur le système","ℹ️")
        col1,col2 = st.columns(2, gap="large")
        with col1:
            section_title("Informations Académiques")
            for label,val in [("Étudiant","Koussay Hassana"),("Encadrant","Tarek Bouchaddekh"),("Établissement","PFE 2025-2026"),("Stage","Amen Bank — Direction des Risques"),("Sujet","Prédiction du risque de crédit par IA")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.7rem 1rem;margin:4px 0;background:white;border-radius:8px;border:1px solid #D1DCE8;border-left:3px solid #002B5C;">
                    <span style="color:#6B7280;font-size:0.82rem;">{label}</span>
                    <span style="color:#002B5C;font-size:0.82rem;font-weight:600;">{val}</span>
                </div>""", unsafe_allow_html=True)
        with col2:
            section_title("Stack Technologique")
            g1,g2 = st.columns(2)
            techs = [("🐍 Python 3.10","#004A99"),("🤖 XGBoost","#002B5C"),("🧠 TensorFlow","#C8102E"),("🔍 SHAP","#B8941F"),("📊 Scikit-learn","#1A7A4A"),("🌐 Streamlit","#6B21A8"),("📈 Matplotlib","#004A99"),("⚖️ SMOTE","#002B5C")]
            for i,(tech,color) in enumerate(techs):
                with (g1 if i%2==0 else g2):
                    st.markdown(f"""<div style="background:white;border:1px solid #D1DCE8;border-radius:8px;padding:0.6rem 0.8rem;margin:4px 0;border-left:3px solid {color};">
                        <span style="color:{color};font-size:0.78rem;font-weight:600;">{tech}</span></div>""", unsafe_allow_html=True)

        divider()
        section_title("Méthodologie du Projet")
        steps=[("01","Exploration (EDA)","Analyse statistique et visualisation du German Credit Dataset (1000 clients, 21 variables)","#002B5C"),
               ("02","Preprocessing","Encodage LabelEncoder, normalisation StandardScaler, équilibrage SMOTE (560/560)","#004A99"),
               ("03","Modélisation","Comparaison de 4 algorithmes ML + réseau de neurones (128-64-32-1)","#C8102E"),
               ("04","Optimisation","GridSearchCV avec validation croisée 5-fold sur 324 combinaisons","#B8941F"),
               ("05","Interprétabilité","Analyse SHAP — waterfall plots, importance globale des variables","#1A7A4A"),
               ("06","Déploiement","Application web Streamlit avec login, tableau de bord et historique","#6B21A8")]
        cols=st.columns(3, gap="medium")
        for i,(num,title,desc,color) in enumerate(steps):
            with cols[i%3]:
                st.markdown(f"""<div style="background:white;border:1px solid #D1DCE8;border-radius:10px;padding:1.2rem;margin-bottom:1rem;border-top:3px solid {color};">
                    <div style="font-family:'Montserrat',sans-serif;font-size:1.5rem;font-weight:800;color:{color};opacity:0.4;">{num}</div>
                    <div style="font-family:'Montserrat',sans-serif;font-weight:700;color:#002B5C;margin:0.3rem 0;font-size:0.9rem;">{title}</div>
                    <div style="font-size:0.76rem;color:#6B7280;line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)