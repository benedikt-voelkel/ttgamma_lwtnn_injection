//////////////////////////////
//Joshua.Wyatt.Smith@cern.ch//
//////////////////////////////
#include "nnInjector.h"
#include <iostream>
#include <memory>

using namespace std;


void m_nan_cleaner_upper(vector<float> *variable){
  for(uint i = 0; i < variable->size(); i++){
    std::cout << variable->at(i) <<std::endl;
  }
}

void m_add_branches(string path,string new_eos_path, string channel, string filename){
  string file = path+channel+"/"+filename;
  string newpath = new_eos_path+ channel+"/"+filename;
  std::cout<<channel<<" "<< filename<< std::endl;

  std::shared_ptr<TFile> newfile(new TFile((newpath.c_str()), "update"));
  // TFile *newfile;
  TTree *newtree=nullptr;

  TChain *fChain=nullptr;
  if (filename.find("QCDfakes") != std::string::npos) {
    fChain = new TChain("nominal_Loose");
  }
  else {
    fChain = new TChain("nominal");
  }

  fChain->Add((file).c_str());

  int nentries = fChain->GetEntries();
  newtree = fChain->CloneTree(0);
  newtree->SetName("nominal");


  // Define some new branches and activate all others //
 //  ///////////////////// For multiclass ////////////////////////
  // newtree->Branch("ph_ISR_MVA",&m_ph_ISR_MVA,"m_ph_ISR_MVA/F");
  // newtree->Branch("ph_FSR_MVA",&m_ph_FSR_MVA,"m_ph_FSR_MVA/F");
  // newtree->Branch("ph_HFake_MVA",&m_ph_HFake_MVA,"m_ph_HFake_MVA/F");
  // newtree->Branch("ph_eFake_MVA",&m_ph_eFake_MVA,"m_ph_eFake_MVA/F");
  // newtree->Branch("ph_OtherPrompt_MVA",&m_ph_OtherPrompt_MVA,"m_ph_OtherPrompt_MVA/F");   
  //////////////////For Prompt photon tagger ////////////////////
    // newtree->Branch("ph_PPT_MVA",&m_ph_PPT_MVA);   
  //////////////////For Prompt photon tagger ////////////////////
  // newtree->Branch("event_ELT_MVA",&m_event_ELT_MVA);   
  ////////////////// Branches sorted by btag weight ////////////////////
  newtree->Branch("jet_tagWeightBin_leading",&jet_tagWeightBin_leading);   
  newtree->Branch("jet_tagWeightBin_subleading",&jet_tagWeightBin_subleading);   
  newtree->Branch("jet_tagWeightBin_subsubleading",&jet_tagWeightBin_subsubleading); 
  //////////////////////////////////////////////////////

  activateBranches(fChain);

  string jetpt;
  string jeteta;
  string jetphi;
  string jetmv2c10;
  string jete;

  string phHFTMVA;
  string phpt;
  string phphi;
  string phe;
  string phcaloeta;
  string phdrleadjet;
  string phdrsubljet;
  string phdrlept;
  string phmgammalept;
  string phmgammaleptlept;

  for (int event = 0; event < nentries; event++) {
    fChain->GetEntry(event);
    loadBar(event, nentries, 100, 50);
    if(jet_pt->size()==0) {continue;}
    if(ph_pt->size()==0) {continue;}
    //m_nan_cleaner_upper(ph_HFT_MVA);

    // Sort btag weigths and add to branch // 
    /////////////////////////////////////////
    std::sort (jet_tagWeightBin->begin(), jet_tagWeightBin->end(), std::greater<int>()); 

    for (uint sorted = 0; sorted < jet_tagWeightBin->size(); sorted++) {
      try {
        jet_tagWeightBin_leading = jet_tagWeightBin->at(0);
      } catch(const std::out_of_range& oor) {
        continue;
      }
      try {
      jet_tagWeightBin_subleading = jet_tagWeightBin->at(1);
      } catch(const std::out_of_range& oor) {
        continue;
      }
     try {
      jet_tagWeightBin_subsubleading = jet_tagWeightBin->at(2);
      } catch(const std::out_of_range& oor) {
        continue;
      }
    }
    //////////////////////////////////////////////

    //----------- Neural network
    m_NeuralNet_input_values["jet_tagWeightBin_0"] = jet_tagWeightBin_leading;
    m_NeuralNet_input_values["jet_tagWeightBin_1"] = jet_tagWeightBin_subleading;
    m_NeuralNet_input_values["jet_tagWeightBin_2"] = jet_tagWeightBin_subsubleading;
    
    // for(uint jet = 0; jet < 5; jet++){
       
    //   jetpt = "jet_pt_"+std::to_string(jet);
    //   jeteta = "jet_eta_"+std::to_string(jet);
    //   jetphi = "jet_phi_"+std::to_string(jet);
    //   jetmv2c10 = "jet_mv2c10_"+std::to_string(jet);
    //   jete = "jet_e_"+std::to_string(jet);

    //   if(jet <= jet_pt->size()-1){
    //     m_NeuralNet_input_values[jetpt] = jet_pt->at(jet);
    //     m_NeuralNet_input_values[jeteta] = jet_eta->at(jet);
    //     m_NeuralNet_input_values[jetphi] = jet_phi->at(jet);
    //     m_NeuralNet_input_values[jetmv2c10] = jet_mv2c10->at(jet);
    //     m_NeuralNet_input_values[jete] = jet_e->at(jet);
    //   }
    //  else{
    //     m_NeuralNet_input_values[jetpt] = 0;
    //     m_NeuralNet_input_values[jeteta] = 0;
    //     m_NeuralNet_input_values[jetphi] = 0;
    //     m_NeuralNet_input_values[jetmv2c10] = 0;
    //     m_NeuralNet_input_values[jete] = 0;
    //   }
    // }
    // m_NeuralNet_input_values["event_mwt"] = event_mwt;
    // m_NeuralNet_input_values["event_nbjets77"] = event_nbjets77;
    // m_NeuralNet_input_values["met_met"] = met_met;
    // m_NeuralNet_input_values["met_phi"] = met_phi;
    // m_NeuralNet_input_values["event_HT"] = event_HT;
    // m_NeuralNet_input_values["event_njets"] = event_njets;
    // m_NeuralNet_input_values["event_mll"] = event_mll;



    // // Photon variabes
    // for(uint photon = 0; photon < 1; photon++){
    //   if(ph_pt->size()==0) {continue;}
    //   phHFTMVA = "ph_HFT_MVA_"+std::to_string(photon);
    //   phpt = "ph_pt_"+std::to_string(photon);
    //   phphi = "ph_phi_"+std::to_string(photon);
    //   phe = "ph_e_"+std::to_string(photon);
    //   phcaloeta = "ph_caloEta_"+std::to_string(photon);
    //   phdrleadjet = "ph_drleadjet_"+std::to_string(photon);
    //   phdrsubljet = "ph_drsubljet_"+std::to_string(photon);

    //   phdrlept = "ph_drlept_"+std::to_string(photon);
    //   phmgammalept = "ph_mgammalept_"+std::to_string(photon);
    //   phmgammaleptlept = "ph_mgammaleptlept_"+std::to_string(photon);

    //   if(photon <= ph_pt->size()-1) {
    //     m_NeuralNet_input_values[phHFTMVA] = ph_HFT_MVA->at(photon);
    //     m_NeuralNet_input_values[phpt] = ph_pt->at(photon);
    //     m_NeuralNet_input_values[phphi] = ph_phi->at(photon);
    //     m_NeuralNet_input_values[phe] = ph_e->at(photon);
    //     m_NeuralNet_input_values[phcaloeta] = ph_caloEta->at(photon);
    //     m_NeuralNet_input_values[phdrleadjet] = ph_drleadjet->at(photon);
    //     m_NeuralNet_input_values[phdrsubljet] = ph_drsubljet->at(photon);

    //     m_NeuralNet_input_values[phdrlept] = ph_drlept->at(photon);
    //     m_NeuralNet_input_values[phmgammalept] = ph_mgammalept->at(photon);
    //     m_NeuralNet_input_values[phmgammaleptlept] = ph_mgammaleptlept->at(photon);
    //   }
    //  else{
    //     m_NeuralNet_input_values[phHFTMVA] = 0;
    //     m_NeuralNet_input_values[phpt] = 0;
    //     m_NeuralNet_input_values[phphi] = 0;
    //     m_NeuralNet_input_values[phe] = 0;
    //     m_NeuralNet_input_values[phcaloeta] = 0;
    //     m_NeuralNet_input_values[phdrleadjet] = 0;
    //     m_NeuralNet_input_values[phdrsubljet] = 0;
    //     m_NeuralNet_input_values[phdrlept] = 0;
    //     m_NeuralNet_input_values[phmgammalept] = 0;
    //     m_NeuralNet_input_values[phmgammaleptlept] = 0;
    //   }
    // }

  //   for (UInt_t pht = 0; pht < ph_pt->size(); pht++) {
    //  m_ph_PPT_MVA[pht]=-99;
  //    // shower shapes with photonID ntuples
    //  m_NeuralNet_input_values["y_Reta"] = ph_reta->at(pht);
    //  m_NeuralNet_input_values["y_Rphi"] = ph_rphi->at(pht);
    //  m_NeuralNet_input_values["y_Rhad"] = ph_rhad->at(pht);
    //  m_NeuralNet_input_values["y_weta1"] = ph_ws3->at(pht);
    //  m_NeuralNet_input_values["y_fracs1"] = ph_fracm->at(pht);
    //  m_NeuralNet_input_values["y_weta2"] = ph_weta2->at(pht);

    //  auto out_vals = m_NeuralNet->compute(m_NeuralNet_input_values);
    //  for (const auto& out: out_vals) {
    //    m_ph_PPT_MVA[pht] = out.second;
    //  }
    // }

    // Fill the tree before calculating NN
    newtree->Fill();
    // Calculate lwtnn NN output
    // auto out_vals = m_NeuralNet->compute(m_NeuralNet_input_values);
    // for (const auto& out: out_vals) {
    //   m_event_ELT_MVA = out.second;
    // }
  }// end event loop

  newfile->cd();
  newtree->Write();
  newfile->Close();

}// end add_nn loop

int main(int argc, char** argv)
{
  gROOT->ProcessLine( "gErrorIgnoreLevel = kError;");
  std::cout << "Found " << argc-1 << " files to run over:" << std::endl;
  std::string in_file_name("../json/lwtnn_EventLevel.json");
  std::ifstream in_file(in_file_name);
  if(!in_file){
    std::cout<<"Error: no nn input file!"<< std::endl;
  }

  lwt::JSONConfig  m_config_netFile = lwt::parse_json(in_file);
  std::cout << "Neural Network has " << m_config_netFile.layers.size() << " layers"<< std::endl;
  m_NeuralNet = new lwt::LightweightNeuralNetwork(m_config_netFile.inputs, 
  m_config_netFile.layers, m_config_netFile.outputs);


  // path to ntuples from AnalysisTop
  string path = "/eos/atlas/user/j/jwsmith/reprocessedNtuples/v007/QE2/";
  string channels[] ={"mujets","ejets"};
  string myPath = "/eos/atlas/user/j/jwsmith/reprocessedNtuples/v007_btagVar/QE2/";

  for (int i = 1; i < argc; ++i) {
    for(const string &c : channels){
      m_add_branches(path,myPath,c,argv[i]);
    }
  }


  return 0;
}
