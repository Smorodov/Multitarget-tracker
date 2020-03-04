#include "ShortPathCalculator.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

#include "muSSP/Graph.h"

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

    std::vector<node> nodes(N + M);

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        nodes[i] = G.new_node();
    }

    edge_map<int> weights(G, 100);
    for (size_t i = 0; i < N; i++)
    {
        bool hasZeroEdge = false;

        for (size_t j = 0; j < M; j++)
        {
            track_t currCost = costMatrix[i + j * N];

            edge e = G.new_edge(nodes[i], nodes[N + j]);

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

    edges_t L = MAX_WEIGHT_BIPARTITE_MATCHING(G, weights);
    for (edges_t::iterator it = L.begin(); it != L.end(); ++it)
    {
        node a = it->source();
        node b = it->target();
        assignment[b.id()] = static_cast<assignments_t::value_type>(a.id() - N);
    }
}

///
/// \brief FindSP
/// \param orgGraph
///
void FindSP(Graph& orgGraph, std::vector<track_t>& pathCost, std::vector<std::vector<int>>& pathSet)
{
    // 1: remove dummy edges
    orgGraph.invalid_edge_rm();

    int path_num = 0;

    // 2: initialize shortest path tree from the DAG
    orgGraph.shortest_path_dag();

    pathCost.push_back(orgGraph.distance2src[orgGraph.sink_id_]);
    orgGraph.cur_path_max_cost = -orgGraph.distance2src[orgGraph.sink_id_]; // the largest cost we can accept

    // 3: convert edge cost (make all weights positive)
    orgGraph.update_allgraph_weights();

    // 8: extract shortest path
    orgGraph.extract_shortest_path();

    pathSet.push_back(orgGraph.shortest_path);
    path_num++;

    std::vector<unsigned long> update_node_num;

    // 4: find nodes for updating based on branch node
    std::vector<int> node_id4updating;
    orgGraph.find_node_set4update(node_id4updating);

    // 10: rebuild residual graph by flipping paths
    orgGraph.flip_path();//also erase the top sinker
    for (;;)
    {
        // 6: update shortest path tree based on the selected sub-graph
        orgGraph.update_shortest_path_tree_recursive(node_id4updating);
        //printf("Iteration #%d, updated node number  %ld \n", path_num, graph.upt_node_num);

        // 7: update sink node (heap)
        orgGraph.update_sink_info(node_id4updating);
        update_node_num.push_back(node_id4updating.size());

        // 8: extract shortest path
        orgGraph.extract_shortest_path();

        // test if stop
        double cur_path_cost = pathCost[path_num - 1] + orgGraph.distance2src[orgGraph.sink_id_];
        if (cur_path_cost > -0.0000001)
            break;

        pathCost.push_back(cur_path_cost);
        orgGraph.cur_path_max_cost = -cur_path_cost;
        pathSet.push_back(orgGraph.shortest_path);
        path_num++;

        // 9: update weights
        orgGraph.update_subgraph_weights(node_id4updating);

        // 4: find nodes for updating
        orgGraph.find_node_set4update(node_id4updating);

        // 10: rebuild the graph
        orgGraph.flip_path();
    }
}

///
/// \brief SPmuSSP::Solve
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
/// \param maxCost
///
void SPmuSSP::Solve(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t /*maxCost*/)
{
    // Add new "layer" to the graph
    if (m_detects.size() < 2)
    {
        m_detects.resize(2);
        m_detects[0].Resize(N);
        m_detects[1].Resize(M);
    }
    else
    {
        assert(m_detects.back().Size() == N);
        m_detects.push_back(Layer());
        m_detects.back().Resize(M);

        if (m_detects.size() > m_settings.m_maxHistory)
            m_detects.pop_front();
    }

    auto layer = m_detects.begin() + m_detects.size() - 1;
    for (size_t i = 0; i < N; ++i)
    {
        Node& node = (*layer)[i];

        for (size_t j = 0; j < M; ++j)
        {
            track_t currCost = costMatrix[i + j * N];
            if (currCost < m_settings.m_distThres)
            {
                node.Add(j, currCost);
                layer->m_arcsCount++;
            }
        }
    }

    // Calc number of nodes and arcs
    int nNodes = 0; // no of nodes
    int nArcs = 0;  // no of arcs
    for (const auto& layer : m_detects)
    {
        nNodes += layer.Size();
        nArcs += layer.m_arcsCount;
    }

    // Create Graph
    Graph orgGraph(nNodes, nArcs, 0, nNodes - 1, 0, 0);
    int edgeID = 0;
    int edgesSum = 0;
    int nodesSum = 0;
    for (const auto& layer : m_detects)
    {
        for (size_t j = 0; j < layer.m_nodes.size(); ++j)
        {
            const auto& node = layer.m_nodes[j];
            for (size_t i = 0; i < node.m_arcs.size(); ++i)
            {
                const auto& arc = node.m_arcs[i];
                int tail = nodesSum + j;
                int head = nodesSum + layer.m_nodes.size() + arc.first;
                orgGraph.add_edge(tail, head, edgeID, arc.second);
                ++edgeID;
            }
        }
        edgesSum += layer.m_arcsCount;
        nodesSum += layer.m_nodes.size();
    }

    // Find paths
    std::vector<track_t> pathCost;
    std::vector<std::vector<int>> pathSet;
    FindSP(orgGraph, pathCost, pathSet);

    track_t costSum = 0;
    for (auto &&i : pathCost)
    {
        costSum += i;
    }
    // printf("The number of paths: %ld, total cost is %.7f, final path cost is: %.7f.\n", path_cost.size(), cost_sum, path_cost[path_cost.size() - 1]);
    // print_solution(org_graph.get(), path_set, "output.txt");//"output_edge_rm.txt"

    auto GetRowIndFromID = [&](int id)
    {
        int res = -1;
        size_t nodesSum = 0;
        for (const auto& layer : m_detects)
        {
            if (nodesSum + layer.m_nodes.size() > static_cast<size_t>(id))
            {
                res = id - nodesSum;
            }
            nodesSum += layer.m_nodes.size();
        }
        return res;
    };
    auto GetRegionIndFromID = [&](int id)
    {
        int res = -1;
        size_t nodesSum = 0;
        for (size_t i = 0; i < m_detects.size(); ++i)
        {
            const auto& layer = m_detects[i];
            if (nodesSum + layer.m_nodes.size() > static_cast<size_t>(id))
            {
                if (i + 1 == m_detects.size())
                    res = id - nodesSum;
                break;
            }
            nodesSum += layer.m_nodes.size();
        }
        return res;
    };

    for (size_t i = 0; i < pathSet.size(); ++i)
    {
        const auto& path = pathSet[i];

        std::map<int, size_t> freq;

        for (size_t j = 0; j < path.size(); ++j)
        {
            int row = GetRowIndFromID(path[j]);
            assert(row >= 0);
            freq[row]++;
        }
        int maxRow = -1;
        size_t maxvals = 0;
        for (auto it : freq)
        {
            if (maxvals < it.second)
            {
                maxvals = it.second;
                maxRow = it.first;
            }
        }
        assert(maxRow >= 0);
        assignment[maxRow] = GetRegionIndFromID(path.size() - 1);
    }
}
