//////////////////////////////
//Joshua.Wyatt.Smith@cern.ch//
//////////////////////////////
#include "nnInjector_dynamic.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional/optional.hpp>

using namespace std;



NNInjector::NNInjector( lwt::LightweightNeuralNetwork* nn ) {
  
  m_nn = nn;
  // Check whether a chain has been initialized.
  m_is_initialized = false;
  m_have_vector = false;
  m_have_scalar = false;

  // So far we just assume that the output directory exists...
  // @todo Add a test for that as well.


}

void NNInjector::read_json( const std::string & json_file_path ) {
  
  // Leave if file cannot be found.
  if( ! std::ifstream( json_file_path ) ) {
    std::cout << "ERROR: File ' " << json_file_path << "' does not exist" << std::endl;
    std::cout << "Leaving...";
    exit(1);
  }

  try{
    // Try to read the JSON file.
    boost::property_tree::ptree pt;
    boost::property_tree::read_json( json_file_path, pt );
    
    for( boost::property_tree::ptree::value_type & branch : pt.get_child( "feature_branches" ) ) {
      // Fill branch map.
      m_branch_map[ branch.first ] = branch.second.data();
    }

    boost::optional< boost::property_tree::ptree& > child = pt.get_child_optional( "more_scalar_branches" );
    if( child ) {
      for( boost::property_tree::ptree::value_type & weight : pt.get_child( "more_scalar_branches" ) ) {
        m_branch_weights[ weight.second.data() ] = -999.9;
      }
    }
    child = pt.get_child_optional( "weights" );
    if( child ) {
      for( boost::property_tree::ptree::value_type & weight : pt.get_child( "weights" ) ) {
        m_branch_weights[ weight.second.data() ] = -999.9;
      }
    }


  } catch(...) {
    std::cout << "ERROR: Branches could not be read from file '" << json_file_path << "'." << std::endl;
    std::cout << "Leaving...";
    exit(1);
  }

}

// Print all branches.
void NNInjector::print_branches() const {
  for( auto const & s : m_branch_map ) {
    std::cout << s.second << std::endl;
  }
}

void NNInjector::initialize( const std::vector< std::string >& input_root_paths, const std::string& treename, const std::string& json_file_path ) {

  read_json( json_file_path );
  m_treename = treename;
  m_input_root_paths = input_root_paths;
  // @note Sort files such that we can compare trees after injection.
  std::sort( m_input_root_paths.begin(), m_input_root_paths.end() );
  
  m_chain = new TChain( m_treename.c_str() );
  for( const auto & f : m_input_root_paths ) {
    m_chain->Add( f.c_str() );
  }
  for( const auto & b : m_branch_map ) {

    TBranch* branch = m_chain->GetBranch( b.second.c_str() );
    TLeaf* leaf = branch->GetLeaf( b.second.c_str() );
    std::string type = leaf->GetTypeName();
    std::cout << "Type of branch " << b.second << " is: " << type << std::endl;

    if( type.find( "vector" ) != std::string::npos ) {

      m_have_vector = true;
      if( type.find( "float" ) != std::string::npos ) {
        m_branches_vector_float[ b.second ] = nullptr;
        //m_branches_vector[ b.second ] = m_branches_vector_float[b.second];
        m_chain->SetBranchAddress( b.second.c_str(), &m_branches_vector_float[b.second] );
        m_branch_vector_float[ b.first ] =  b.second;
      } else if( type.find( "int" ) != std::string::npos ) {
        m_branches_vector_int[ b.second ] = nullptr;
        // m_branches_vector[ b.second ] = m_branches_vector_int[b.second];
        m_chain->SetBranchAddress( b.second.c_str(), &m_branches_vector_int[b.second] );
        m_branch_vector_int[ b.first ] =  b.second;
      } else {
        std::cout << "ERROR: Cannot find matching type for branch type " << type << "." << std::endl;
        exit(1);
      }

    } else if( type.find( "Int" ) != std::string::npos ) {
      m_branches_scalar_int[ b.second ] = -9999;
      m_chain->SetBranchAddress( b.second.c_str(), &m_branches_scalar_int[b.second] );
      // m_branches_scalar[b.second ] = m_branches_scalar_int[b.second];
      m_branch_vector_int[ b.first ] = b.second;
      m_have_scalar = true;
    } else if( type.find( "Float" ) != std::string::npos ) {
      m_branches_scalar_float[ b.second ] = -9999.9;
      m_chain->SetBranchAddress( b.second.c_str(), &m_branches_scalar_float[b.second] );
      // m_branches_scalar[b.second ] = m_branches_scalar_float[b.second];
      m_branch_vector_float[ b.first ] = b.second;
      m_have_scalar = true;
    } else {
      std::cout << "ERROR: Cannot find matching type for branch type " << type << "." << std::endl;
      exit(1);
    }
    // Make sure we have only vectors or scalars...
    if( m_have_scalar && m_have_vector ) {
      std::cout << "ERROR: EITHER scalar OR vector branches." << std::endl;
      exit(1);
    }

  }

  for( auto const & wb : m_branch_weights ) {
    m_chain->SetBranchAddress( wb.first.c_str(), &m_branch_weights[wb.first] );
  }
  m_is_initialized = true;
}

void NNInjector::inject( const std::string& output_root_path ) {
  inject( output_root_path, "nominal", "NN_output" );
}

void NNInjector::inject( const std::string& output_root_path, const std::string & new_tree_name, const std::string & new_branch_name) {

  if( !m_is_initialized ) {
    std::cout << "ERROR: Not initialized!" << std::endl;
    exit(1);
  }

  // Need the output file and tree.
  TFile* output_root = new TFile( output_root_path.c_str(), "recreate" );
  TTree* output_tree = new TTree( new_tree_name.c_str(), new_tree_name.c_str() );
  
  // Set branches. We need to do this specifically since we might mix vectors (from input) with scalars if we only clone the initial chain.
  std::map< std::string, double > copied_branches;
  for( auto const & b : m_branch_map ) {
    copied_branches[b.second] = -999.9;
    output_tree->Branch( b.second.c_str(), &copied_branches[b.second] );
  }

  for( auto const & wb : m_branch_weights ) {
    output_tree->Branch( wb.first.c_str(), &m_branch_weights[wb.first] );
  }
  

  // Variable to store NN response.
  double NN_output = -999;
  // Set new Branch in output tree.
  output_tree->Branch( new_branch_name.c_str(), &NN_output );

  
  std::map< std::string, double > nn_input_values;
  
  unsigned int n_entries = m_chain->GetEntries();
  
  std::cout << "+-----------+" << std::endl;
  std::cout << "| INJECTING |" << std::endl;
  std::cout << "+-----------+" << std::endl;
  for( unsigned int i = 0; i < n_entries; i++ ) {
    
    m_chain->GetEntry( i );

    // Loop over branches we have.
    if( i % 100000 == 0 ) {
      std::cout << "Done with " << i << " events..." << std::endl;
    } else if( i % 10000 == 0 && i < 100000 ) {
      std::cout << "Done with " << i << " events..." << std::endl;
    }

    if( m_have_vector ) {
      unsigned int size = 0;
      try {
        size = m_branches_vector_float.begin()->second->size();
      } catch(...) {
        size = m_branches_vector_int.begin()->second->size();
      }

      // Assume that all branches have same length.
      for( unsigned int i = 0; i < size; ++i ) {
        // Loop over branch map and fill nn values.
        for( const auto & branch : m_branch_vector_float ) {
          nn_input_values[ branch.first ] = m_branches_vector_float[branch.second]->at(i);
          // Fill copied branches explicitly to flatten the stuff in case of vectors in input tree.
          copied_branches[branch.second] = m_branches_vector_float[branch.second]->at(i);
        }
        for( const auto & branch : m_branch_vector_int ) {
          nn_input_values[ branch.first ] = m_branches_vector_int[branch.second]->at(i);
          // Fill copied branches explicitly to flatten the stuff in case of vectors in input tree.
          copied_branches[branch.second] = m_branches_vector_int[branch.second]->at(i);
        }
        auto out_vals = m_nn->compute( nn_input_values );
        // @todo Why do we need this loop? Cross-check with Josh!
        for( const auto & out : out_vals ) {
          NN_output = out.second;
        }
        // Fill for each entry in vector.
        output_tree->Fill();

      }
      // We have scalar input branches.
    } else {

      for( const auto & branch : m_branch_vector_float ) {
        nn_input_values[ branch.first ] = m_branches_scalar_float[branch.second];
        copied_branches[branch.second] = m_branches_scalar_float[branch.second];
      }
      for( const auto & branch : m_branch_vector_int ) {
        nn_input_values[ branch.first ] = m_branches_scalar_int[branch.second];
        copied_branches[branch.second] = m_branches_scalar_int[branch.second];
      }

      auto out_vals = m_nn->compute( nn_input_values );
      // @todo Why do we need this loop? Cross-check with Josh!
      for( const auto & out : out_vals ) {
        NN_output = out.second;
      }

      output_tree->Fill();

    } // end else
  } // end loop over entries.
  output_tree->Write();
  output_root->Close();
  std::cout << "+----------------+" << std::endl;
  std::cout << "| INJECTING DONE |" << std::endl;
  std::cout << "+----------------+" << std::endl;

}


void m_nan_cleaner_upper(vector<float> *variable){
  for(uint i = 0; i < variable->size(); i++){
    std::cout << variable->at(i) <<std::endl;
  }
}
