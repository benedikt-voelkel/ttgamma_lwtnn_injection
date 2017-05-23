//////////////////////////////
//Joshua.Wyatt.Smith@cern.ch//
//////////////////////////////
#include <iostream>
#include <string>
#include <tuple>
#include <fstream>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <string>
#include <fstream>
#include "TCut.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include <string>
#include <TChainElement.h>
#include "TH1D.h"
#include "TH1F.h"
#include "TSystemDirectory.h"
#include "TSystemFile.h"
#include <sstream>
#include "TLorentzVector.h"
#include <memory>
#include <typeinfo>
#include "TGraph.h"
#include "lwtnn/NNLayerConfig.hh"
#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"
#include "lwtnn/Stack.hh" // <-- added for exceptions
using namespace std;

//void m_add_nn();
////////For multiclass /////////////
// std::vector<float> m_ph_ISR_MVA;
// std::vector<float> m_ph_FSR_MVA;
// std::vector<float> m_ph_HFake_MVA;
// std::vector<float> m_ph_eFake_MVA;
// std::vector<float> m_ph_OtherPrompt_MVA;
/////////////////////////////////////////////
////////For PPT /////////////
float m_event_ELT_MVA;
/////////////////////////////////////////////
lwt::JSONConfig m_config_netFile;
std::map<std::string,double> m_NeuralNet_input_values;
lwt::LightweightNeuralNetwork *m_NeuralNet;
/////////////////////////////////////////////


Float_t         weight_mm_ejets;
Float_t         weight_mm_mujets;
Float_t         weight_mc;
Float_t         weight_pileup;
Float_t         weight_leptonSF;
Float_t         weight_photonSF;
Float_t         weight_bTagSF_77;
Float_t         weight_bTagSF_85;
Float_t         weight_bTagSF_70;
Float_t         weight_bTagSF_Continuous;
Float_t         weight_jvt;
Float_t         weight_pileup_UP;
Float_t         weight_pileup_DOWN;
ULong64_t       eventNumber;
UInt_t          runNumber;
UInt_t          randomRunNumber;
UInt_t          mcChannelNumber;
Float_t         mu;
UInt_t          backgroundFlags;
UInt_t          hasBadMuon;
vector<float>   *el_pt = nullptr;
vector<float>   *el_eta = nullptr;
vector<float>   *el_cl_eta = nullptr;
vector<float>   *el_phi = nullptr;
vector<float>   *el_e = nullptr;
vector<float>   *el_charge = nullptr;
vector<float>   *el_topoetcone20 = nullptr;
vector<float>   *el_ptvarcone20 = nullptr;
vector<char>    *el_CF = nullptr;
vector<float>   *el_d0sig = nullptr;
vector<float>   *el_delta_z0_sintheta = nullptr;
vector<int>     *el_true_type = nullptr;
vector<int>     *el_true_origin = nullptr;
vector<int>     *el_true_typebkg = nullptr;
vector<int>     *el_true_originbkg = nullptr;
vector<float>   *mu_pt= nullptr;
vector<float>   *mu_eta= nullptr;
vector<float>   *mu_phi= nullptr;
vector<float>   *mu_e= nullptr;
vector<float>   *mu_charge= nullptr;
vector<float>   *mu_topoetcone20= nullptr;
vector<float>   *mu_ptvarcone30= nullptr;
vector<float>   *mu_d0sig= nullptr;
vector<float>   *mu_delta_z0_sintheta= nullptr;
vector<int>     *mu_true_type= nullptr;
vector<int>     *mu_true_origin= nullptr;
vector<float>   *ph_pt= nullptr;
vector<float>   *ph_eta= nullptr;
vector<float>   *ph_phi= nullptr;
vector<float>   *ph_e= nullptr;
vector<float>   *ph_iso= nullptr;
vector<float>   *jet_pt= nullptr;
vector<float>   *jet_eta= nullptr;
vector<float>   *jet_phi= nullptr;
vector<float>   *jet_e= nullptr;
vector<float>   *jet_mv2c00= nullptr;
vector<float>   *jet_mv2c10= nullptr;
vector<float>   *jet_mv2c20= nullptr;
vector<float>   *jet_ip3dsv1= nullptr;
vector<float>   *jet_jvt= nullptr;
vector<char>    *jet_passfjvt= nullptr;
vector<int>     *jet_truthflav= nullptr;
vector<int>     *jet_truthPartonLabel= nullptr;
vector<char>    *jet_isTrueHS= nullptr;
vector<char>    *jet_isbtagged_77= nullptr;
vector<char>    *jet_isbtagged_85= nullptr;
vector<char>    *jet_isbtagged_70= nullptr;
vector<int>     *jet_tagWeightBin= nullptr;
float           jet_tagWeightBin_leading;
float           jet_tagWeightBin_subleading;
float           jet_tagWeightBin_subsubleading;
Float_t         met_met;
Float_t         met_phi;
Int_t           ejets_2015;
Int_t           ejets_2016;
Int_t           mujets_2015;
Int_t           mujets_2016;
Int_t           emu_2015;
Int_t           emu_2016;
Int_t           emuSS_2015;
Int_t           emuSS_2016;
Int_t           ee_2015;
Int_t           ee_2016;
Int_t           eeSS_2015;
Int_t           eeSS_2016;
Int_t           mumu_2015;
Int_t           mumu_2016;
Int_t           mumuSS_2015;
Int_t           mumuSS_2016;
Char_t          HLT_mu20_iloose_L1MU15;
Char_t          HLT_e60_lhmedium_nod0;
Char_t          HLT_mu26_ivarmedium;
Char_t          HLT_e26_lhtight_nod0_ivarloose;
Char_t          HLT_e140_lhloose_nod0;
Char_t          HLT_mu50;
Char_t          HLT_e60_lhmedium;
Char_t          HLT_e24_lhmedium_L1EM20VH;
Char_t          HLT_e120_lhloose;
vector<char>    *el_trigMatch_HLT_e60_lhmedium_nod0= nullptr;
vector<char>    *el_trigMatch_HLT_e26_lhtight_nod0_ivarloose= nullptr;
vector<char>    *el_trigMatch_HLT_e140_lhloose_nod0= nullptr;
vector<char>    *el_trigMatch_HLT_e60_lhmedium= nullptr;
vector<char>    *el_trigMatch_HLT_e24_lhmedium_L1EM20VH= nullptr;
vector<char>    *el_trigMatch_HLT_e120_lhloose= nullptr;
vector<char>    *mu_trigMatch_HLT_mu26_ivarmedium= nullptr;
vector<char>    *mu_trigMatch_HLT_mu50= nullptr;
vector<char>    *mu_trigMatch_HLT_mu20_iloose_L1MU15= nullptr;
vector<float>   *ph_topoetcone20= nullptr;
vector<float>   *ph_topoetcone30= nullptr;
vector<float>   *ph_topoetcone40= nullptr;
vector<float>   *ph_ptcone20= nullptr;
vector<float>   *ph_ptcone30= nullptr;
vector<float>   *ph_ptcone40= nullptr;
vector<float>   *ph_ptvarcone20= nullptr;
vector<float>   *ph_ptvarcone30= nullptr;
vector<float>   *ph_ptvarcone40= nullptr;
vector<char>    *ph_isoFCTCO= nullptr;
vector<char>    *ph_isoFCT= nullptr;
vector<char>    *ph_isoFCL= nullptr;
vector<char>    *ph_isTight= nullptr;
vector<char>    *ph_isLoose= nullptr;
vector<char>    *ph_isTight_daod= nullptr;
vector<char>    *ph_isHFake= nullptr;
vector<char>    *ph_isHadronFakeFailedDeltaE= nullptr;
vector<char>    *ph_isHadronFakeFailedFside= nullptr;
vector<char>    *ph_isHadronFakeFailedWs3= nullptr;
vector<char>    *ph_isHadronFakeFailedERatio= nullptr;
vector<float>   *ph_rhad1= nullptr;
vector<float>   *ph_rhad= nullptr;
vector<float>   *ph_reta= nullptr;
vector<float>   *ph_weta2= nullptr;
vector<float>   *ph_rphi= nullptr;
vector<float>   *ph_ws3= nullptr;
vector<float>   *ph_wstot= nullptr;
vector<float>   *ph_fracm= nullptr;
vector<float>   *ph_deltaE= nullptr;
vector<float>   *ph_eratio= nullptr;
vector<float>   *ph_emaxs1= nullptr;
vector<float>   *ph_f1= nullptr;
vector<float>   *ph_e277= nullptr;
vector<unsigned int> *ph_OQ= nullptr;
vector<unsigned int> *ph_author= nullptr;
vector<int>     *ph_conversionType= nullptr;
vector<float>   *ph_caloEta= nullptr;
vector<unsigned int> *ph_isEM= nullptr;
vector<int>     *ph_nVertices= nullptr;
vector<float>   *ph_SF_eff= nullptr;
vector<float>   *ph_SF_effUP= nullptr;
vector<float>   *ph_SF_effDO= nullptr;
vector<float>   *ph_SF_iso= nullptr;
vector<float>   *ph_SF_lowisoUP= nullptr;
vector<float>   *ph_SF_lowisoDO= nullptr;
vector<float>   *ph_SF_trkisoUP= nullptr;
vector<float>   *ph_SF_trkisoDO= nullptr;
vector<float>   *ph_drleadjet= nullptr;
vector<float>   *ph_dralljet= nullptr;
vector<float>   *ph_drsubljet= nullptr;
vector<float>   *ph_drlept= nullptr;
vector<float>   *ph_mgammalept= nullptr;
vector<float>   *ph_mgammaleptlept= nullptr;
Int_t           selph_index1;
Int_t           selph_index2;
Int_t           selhf_index1;
vector<int>     *ph_truthType= nullptr;
vector<int>     *ph_truthOrigin= nullptr;
vector<int>     *ph_truthAncestor= nullptr;
vector<int>     *ph_mc_pid= nullptr;
vector<int>     *ph_mc_barcode= nullptr;
vector<float>   *ph_mc_pt= nullptr;
vector<float>   *ph_mc_eta= nullptr;
vector<float>   *ph_mc_phi= nullptr;
vector<float>   *ph_mcel_dr= nullptr;
vector<float>   *ph_mcel_pt= nullptr;
vector<float>   *ph_mcel_eta= nullptr;
vector<float>   *ph_mcel_phi= nullptr;
vector<float>   *mcph_pt= nullptr;
vector<float>   *mcph_eta= nullptr;
vector<float>   *mcph_phi= nullptr;
vector<int>     *mcph_ancestor= nullptr;
vector<int>     *el_truthAncestor= nullptr;
vector<char>    *el_isoGradient= nullptr;
vector<int>     *el_mc_pid= nullptr;
vector<float>   *el_mc_charge= nullptr;
vector<float>   *el_mc_pt= nullptr;
vector<float>   *el_mc_eta= nullptr;
vector<float>   *el_mc_phi= nullptr;
vector<float>   *mu_mc_charge= nullptr;
vector<float>   *jet_mcdr_tW1= nullptr;
vector<float>   *jet_mcdr_tW2= nullptr;
vector<float>   *jet_mcdr_tB= nullptr;
vector<float>   *jet_mcdr_tbW1= nullptr;
vector<float>   *jet_mcdr_tbW2= nullptr;
vector<float>   *jet_mcdr_tbB= nullptr;
vector<int>     *lepton_type= nullptr;
Float_t         event_HT;
Float_t         event_mll;
Float_t         event_mwt;
Int_t           event_njets;
Int_t           event_nbjets70;
Int_t           event_nbjets77;
Int_t           event_nbjets85;
Int_t           event_ngoodphotons;
Int_t           event_photonorigin;
Int_t           event_photonoriginTA;
Int_t           event_nhadronfakes;
vector<float>   *ph_HFT_MVA= nullptr;
Int_t           ph_nHFT_MVA;
Double_t        event_norm;
Double_t        event_lumi;



//Some new variables
float ISR_MVA = -1;
float FSR_MVA = -1;
float hadronfake_MVA = -1;
float egammafake_MVA = -1;
float other_MVA = -1;

void activateBranches(TChain *chain){
	// Activate branches for MC chain
	#define SETBRANCH(branchName) chain->SetBranchAddress(#branchName,&branchName)
	SETBRANCH(weight_mm_ejets);
	SETBRANCH(weight_mm_mujets);
	SETBRANCH(weight_mc);
	SETBRANCH(weight_pileup);
	SETBRANCH(weight_leptonSF);
	SETBRANCH(weight_photonSF);
	SETBRANCH(weight_bTagSF_77);
	SETBRANCH(weight_bTagSF_85);
	SETBRANCH(weight_bTagSF_70);
	SETBRANCH(weight_bTagSF_Continuous);
	SETBRANCH(weight_jvt);
	SETBRANCH(weight_pileup_UP);
	SETBRANCH(weight_pileup_DOWN);
	SETBRANCH(eventNumber);
	SETBRANCH(runNumber);
	SETBRANCH(randomRunNumber);
	SETBRANCH(mcChannelNumber);
	SETBRANCH(mu);
	SETBRANCH(backgroundFlags);
	SETBRANCH(hasBadMuon);
	SETBRANCH(el_pt);
	SETBRANCH(el_eta);
	SETBRANCH(el_cl_eta);
	SETBRANCH(el_phi);
	SETBRANCH(el_e);
	SETBRANCH(el_charge);
	SETBRANCH(el_topoetcone20);
	SETBRANCH(el_ptvarcone20);
	SETBRANCH(el_CF);
	SETBRANCH(el_d0sig);
	SETBRANCH(el_delta_z0_sintheta);
	SETBRANCH(el_true_type);
	SETBRANCH(el_true_origin);
	SETBRANCH(el_true_typebkg);
	SETBRANCH(el_true_originbkg);
	SETBRANCH(mu_pt);
	SETBRANCH(mu_eta);
	SETBRANCH(mu_phi);
	SETBRANCH(mu_e);
	SETBRANCH(mu_charge);
	SETBRANCH(mu_topoetcone20);
	SETBRANCH(mu_ptvarcone30);
	SETBRANCH(mu_d0sig);
	SETBRANCH(mu_delta_z0_sintheta);
	SETBRANCH(mu_true_type);
	SETBRANCH(mu_true_origin);
	SETBRANCH(ph_pt);
	SETBRANCH(ph_eta);
	SETBRANCH(ph_phi);
	SETBRANCH(ph_e);
	SETBRANCH(ph_iso);
	SETBRANCH(jet_pt);
	SETBRANCH(jet_eta);
	SETBRANCH(jet_phi);
	SETBRANCH(jet_e);
	SETBRANCH(jet_mv2c00);
	SETBRANCH(jet_mv2c10);
	SETBRANCH(jet_mv2c20);
	SETBRANCH(jet_ip3dsv1);
	SETBRANCH(jet_jvt);
	SETBRANCH(jet_passfjvt);
	SETBRANCH(jet_truthflav);
	SETBRANCH(jet_truthPartonLabel);
	SETBRANCH(jet_isTrueHS);
	SETBRANCH(jet_isbtagged_77);
	SETBRANCH(jet_isbtagged_85);
	SETBRANCH(jet_isbtagged_70);
	SETBRANCH(jet_tagWeightBin);
	SETBRANCH(met_met);
	SETBRANCH(met_phi);
	SETBRANCH(ejets_2015);
	SETBRANCH(ejets_2016);
	SETBRANCH(mujets_2015);
	SETBRANCH(mujets_2016);
	SETBRANCH(emu_2015);
	SETBRANCH(emu_2016);
	SETBRANCH(emuSS_2015);
	SETBRANCH(emuSS_2016);
	SETBRANCH(ee_2015);
	SETBRANCH(ee_2016);
	SETBRANCH(eeSS_2015);
	SETBRANCH(eeSS_2016);
	SETBRANCH(mumu_2015);
	SETBRANCH(mumu_2016);
	SETBRANCH(mumuSS_2015);
	SETBRANCH(mumuSS_2016);
	SETBRANCH(HLT_mu20_iloose_L1MU15);
	SETBRANCH(HLT_e60_lhmedium_nod0);
	SETBRANCH(HLT_mu26_ivarmedium);
	SETBRANCH(HLT_e26_lhtight_nod0_ivarloose);
	SETBRANCH(HLT_e140_lhloose_nod0);
	SETBRANCH(HLT_mu50);
	SETBRANCH(HLT_e60_lhmedium);
	SETBRANCH(HLT_e24_lhmedium_L1EM20VH);
	SETBRANCH(HLT_e120_lhloose);
	SETBRANCH(el_trigMatch_HLT_e60_lhmedium_nod0);
	SETBRANCH(el_trigMatch_HLT_e26_lhtight_nod0_ivarloose);
	SETBRANCH(el_trigMatch_HLT_e140_lhloose_nod0);
	SETBRANCH(el_trigMatch_HLT_e60_lhmedium);
	SETBRANCH(el_trigMatch_HLT_e24_lhmedium_L1EM20VH);
	SETBRANCH(el_trigMatch_HLT_e120_lhloose);
	SETBRANCH(mu_trigMatch_HLT_mu26_ivarmedium);
	SETBRANCH(mu_trigMatch_HLT_mu50);
	SETBRANCH(mu_trigMatch_HLT_mu20_iloose_L1MU15);
	SETBRANCH(ph_topoetcone20);
	SETBRANCH(ph_topoetcone30);
	SETBRANCH(ph_topoetcone40);
	SETBRANCH(ph_ptcone20);
	SETBRANCH(ph_ptcone30);
	SETBRANCH(ph_ptcone40);
	SETBRANCH(ph_ptvarcone20);
	SETBRANCH(ph_ptvarcone30);
	SETBRANCH(ph_ptvarcone40);
	SETBRANCH(ph_isoFCTCO);
	SETBRANCH(ph_isoFCT);
	SETBRANCH(ph_isoFCL);
	SETBRANCH(ph_isTight);
	SETBRANCH(ph_isLoose);
	SETBRANCH(ph_isTight_daod);
	SETBRANCH(ph_isHFake);
	SETBRANCH(ph_isHadronFakeFailedDeltaE);
	SETBRANCH(ph_isHadronFakeFailedFside);
	SETBRANCH(ph_isHadronFakeFailedWs3);
	SETBRANCH(ph_isHadronFakeFailedERatio);
	SETBRANCH(ph_rhad1);
	SETBRANCH(ph_rhad);
	SETBRANCH(ph_reta);
	SETBRANCH(ph_weta2);
	SETBRANCH(ph_rphi);
	SETBRANCH(ph_ws3);
	SETBRANCH(ph_wstot);
	SETBRANCH(ph_fracm);
	SETBRANCH(ph_deltaE);
	SETBRANCH(ph_eratio);
	SETBRANCH(ph_emaxs1);
	SETBRANCH(ph_f1);
	SETBRANCH(ph_e277);
	SETBRANCH(ph_OQ);
	SETBRANCH(ph_author);
	SETBRANCH(ph_conversionType);
	SETBRANCH(ph_caloEta);
	SETBRANCH(ph_isEM);
	SETBRANCH(ph_nVertices);
	SETBRANCH(ph_SF_eff);
	SETBRANCH(ph_SF_effUP);
	SETBRANCH(ph_SF_effDO);
	SETBRANCH(ph_SF_iso);
	SETBRANCH(ph_SF_lowisoUP);
	SETBRANCH(ph_SF_lowisoDO);
	SETBRANCH(ph_SF_trkisoUP);
	SETBRANCH(ph_SF_trkisoDO);
	SETBRANCH(ph_drleadjet);
	SETBRANCH(ph_dralljet);
	SETBRANCH(ph_drsubljet);
	SETBRANCH(ph_drlept);
	SETBRANCH(ph_mgammalept);
	SETBRANCH(ph_mgammaleptlept);
	SETBRANCH(selph_index1);
	SETBRANCH(selph_index2);
	SETBRANCH(selhf_index1);
	SETBRANCH(ph_truthType);
	SETBRANCH(ph_truthOrigin);
	SETBRANCH(ph_truthAncestor);
	SETBRANCH(ph_mc_pid);
	SETBRANCH(ph_mc_barcode);
	SETBRANCH(ph_mc_pt);
	SETBRANCH(ph_mc_eta);
	SETBRANCH(ph_mc_phi);
	SETBRANCH(ph_mcel_dr);
	SETBRANCH(ph_mcel_pt);
	SETBRANCH(ph_mcel_eta);
	SETBRANCH(ph_mcel_phi);
	SETBRANCH(mcph_pt);
	SETBRANCH(mcph_eta);
	SETBRANCH(mcph_phi);
	SETBRANCH(mcph_ancestor);
	SETBRANCH(el_truthAncestor);
	SETBRANCH(el_isoGradient);
	SETBRANCH(el_mc_pid);
	SETBRANCH(el_mc_charge);
	SETBRANCH(el_mc_pt);
	SETBRANCH(el_mc_eta);
	SETBRANCH(el_mc_phi);
	SETBRANCH(mu_mc_charge);
	SETBRANCH(jet_mcdr_tW1);
	SETBRANCH(jet_mcdr_tW2);
	SETBRANCH(jet_mcdr_tB);
	SETBRANCH(jet_mcdr_tbW1);
	SETBRANCH(jet_mcdr_tbW2);
	SETBRANCH(jet_mcdr_tbB);
	SETBRANCH(lepton_type);
	SETBRANCH(event_HT);
	SETBRANCH(event_mll);
	SETBRANCH(event_mwt);
	SETBRANCH(event_njets);
	SETBRANCH(event_nbjets70);
	SETBRANCH(event_nbjets77);
	SETBRANCH(event_nbjets85);
	SETBRANCH(event_ngoodphotons);
	SETBRANCH(event_photonorigin);
	SETBRANCH(event_photonoriginTA);
	SETBRANCH(event_nhadronfakes);
	SETBRANCH(ph_HFT_MVA);
	SETBRANCH(ph_nHFT_MVA);
	SETBRANCH(event_norm);
	SETBRANCH(event_lumi);
	#undef SETBRANCH
	}
// A progress bar
static inline void loadBar(int x, int n, int r, int w)
{
	// Only update r times.
	if (x % (n / r + 1) != 0) {
		return;
	}
	// Calculuate the ratio of complete-to-incomplete.
	float ratio = x / (float)n;
	int c = ratio * w;
	// Show the percentage complete.
	printf("%3d%% [", (int)(ratio * 100));
	// Show the load bar.
	for (int x = 0; x < c; x++) {
		printf("=");
	}
	for (int x = c; x < w; x++) {
		printf(" ");
	}
	// ANSI Control codes to go back to the
	// previous line and clear it.
	printf("]\n\033[F\033[J");
}
