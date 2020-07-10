#include "ShortPathCalculator.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

///
/// \brief SPBipart::Solve
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
/// \param maxCost
///
void SPBipart::Solve(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t maxCost)
{
    MyGraph G;
    G.make_directed();

    std::vector<GTL::node> nodes(N + M);

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        nodes[i] = G.new_node();
    }

	GTL::edge_map<int> weights(G, 100);
    for (size_t i = 0; i < N; i++)
    {
        bool hasZeroEdge = false;

        for (size_t j = 0; j < M; j++)
        {
            track_t currCost = costMatrix[i + j * N];

			GTL::edge e = G.new_edge(nodes[i], nodes[N + j]);

            if (currCost < m_settings.m_distThres)
            {
                int weight = static_cast<int>(maxCost - currCost + 1);
                G.set_edge_weight(e, weight);
                weights[e] = weight;
            }
            else
            {
                if (!hasZeroEdge)
                {
                    G.set_edge_weight(e, 0);
                    weights[e] = 0;
                }
                hasZeroEdge = true;
            }
        }
    }

	GTL::edges_t L = MAX_WEIGHT_BIPARTITE_MATCHING(G, weights);
    for (GTL::edges_t::iterator it = L.begin(); it != L.end(); ++it)
    {
        GTL::node a = it->source();
        GTL::node b = it->target();
        assignment[b.id()] = static_cast<assignments_t::value_type>(a.id() - N);
    }
}
