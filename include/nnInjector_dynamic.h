//////////////////////////////
//Joshua.Wyatt.Smith@cern.ch//
//////////////////////////////
#include <iostream>
#include <string>
#include <tuple>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
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
#include "TBranch.h"
#include "TLeaf.h"
#include "TLeafD.h"
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

#include <boost/variant.hpp>
#include <boost/variant/multivisitors.hpp>

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

/////////////////////////////////////////////
// Try dynamic branches with Boost
/////////////////////////////////////////////


class NNInjector {

  public:
    NNInjector( lwt::LightweightNeuralNetwork* );
    NNInjector( NNInjector& ) = delete;
    // @todo Make sure that pointer to chain is deleted?!?!
    ~NNInjector() {}
    NNInjector operator=( NNInjector& ) = delete;

    // Pointer to NN.
    lwt::LightweightNeuralNetwork* m_nn;

    // Input and output ROOT paths.
    std::vector< std::string > m_input_root_paths;
    std::string m_output_root_paths;
    // Input treename
    std::string m_treename;
    
    TChain* m_chain;
    // Check whether chain is initialized.
    bool m_is_initialized;

    // NN input values
    std::map< std::string, double > m_nn_input_values;

    // Path to json map file.
    std::string m_json_file_path;

    // Try to cast everything to vector< float > or to scalar float. Maybe we need
    // int or others as well.vector< boost::variant< float, vector< float >* > > branches;
    //std::map< std::string, boost::variant< float, vector< float >* > > branches;
    std::map< std::string, float > m_branches_scalar_float;
    std::map< std::string, int > m_branches_scalar_int;
    std::map< std::string, std::vector< float >* > m_branches_vector_float;
    std::map< std::string, std::vector< int >* > m_branches_vector_int;
    //std::vector< float > m_branches_scalar_float;
    //std::vector< int > m_branches_scalar_int;
    //std::vector< std::vector< float >* > m_branches_vector_float;
    //std::vector< std::vector< int >* > m_branches_vector_int;

    // Mapping everything into a boost variant.
    // std::map< std::string, boost::variant< float&, int& > > m_branches_scalar;
    // std::map< std::string, boost::variant< std::vector<float>*, std::vector<int>* > > m_branches_vector;


    // std::map to map branch names. E.g. thing was trained on ph_pt but is now applied to a tree with branch ph.pt .
    // Would be branch_map[ "ph_pt" ] = "ph.pt";
    std::map< std::string, std::string > m_branch_map;
    std::map< std::string, std::string > m_branch_vector_int;
    std::map< std::string, std::string > m_branch_vector_float;

    // Weight branches are assumed to be scalars and also floats.
    std::map< std::string, float > m_branch_weights;

    // So far we run EITHER over vector like branches OR scalar ones.
    bool m_have_vector;
    bool m_have_scalar;

      
    // Read JSON file.
    void read_json( const std::string &);
    void print_branches() const;
    
    //bool check_trees() const;
    //
    void initialize( const std::vector< std::string >&, const std::string&, const std::string& );
    
    void inject( const std::string& );
    void inject( const std::string&, const std::string&, const std::string&);



};
/*
class pointer_visitor : public boost::static_visitor<boost::variant< std::vector<float>*, std::vector<int>* > > {

  public:
    boost::variant< std::vector<float>*, std::vector<int*> > operator()( std::vector<float>* v ) {
      return v;
    }
    boost::variant< std::vector<float>*, std::vector<int*> > operator()( std::vector<int>* v ) {
      return v;
    }

};
*/


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
