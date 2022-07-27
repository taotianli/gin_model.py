import os
import json
import time
import argparse
import numpy as np
import dgl
import torch as th
import pyarrow
import pandas as pd
from pyarrow import csv


parser = argparse.ArgumentParser(description='Construct graph partitions')
parser.add_argument('--input-dir', required=True, type=str,
                    help='The directory path that contains the partition results.')
parser.add_argument('--graph-name', required=True, type=str,
                    help='The graph name')
parser.add_argument('--schema', required=True, type=str,
                    help='The schema of the graph')
parser.add_argument('--num-parts', required=True, type=int,
                    help='The number of partitions')
parser.add_argument('--num-node-weights', required=True, type=int,
                    help='The number of node weights used by METIS.')
parser.add_argument('--workspace', type=str, default='/tmp',
                    help='The directory to store the intermediate results')
parser.add_argument('--node-attr-dtype', type=str, default=None,
                    help='The data type of the node attributes')
parser.add_argument('--edge-attr-dtype', type=str, default=None,
                    help='The data type of the edge attributes')
parser.add_argument('--output', required=True, type=str,
                    help='The output directory of the partitioned results')
parser.add_argument('--removed-edges', help='a file that contains the removed self-loops and duplicated edges',
                    default=None, type=str)

args = parser.parse_args()

input_dir = args.input_dir
graph_name = args.graph_name
num_parts = args.num_parts
num_node_weights = args.num_node_weights
node_attr_dtype = args.node_attr_dtype
edge_attr_dtype = args.edge_attr_dtype
workspace_dir = args.workspace
output_dir = args.output

self_loop_edges = None
duplicate_edges = None
if args.removed_edges is not None:
    removed_file = '{}/{}'.format(input_dir, args.removed_edges)
    removed_df = csv.read_csv(removed_file, read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                              parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
    assert removed_df.num_columns == 4
    src_id = removed_df['f0'].to_numpy()
    dst_id = removed_df['f1'].to_numpy()
    orig_id = removed_df['f2'].to_numpy()
    etype = removed_df['f3'].to_numpy()
    self_loop_idx = src_id == dst_id
    not_self_loop_idx = src_id != dst_id
    self_loop_edges = [src_id[self_loop_idx], dst_id[self_loop_idx],
                       orig_id[self_loop_idx], etype[self_loop_idx]]
    duplicate_edges = [src_id[not_self_loop_idx], dst_id[not_self_loop_idx],
                       orig_id[not_self_loop_idx], etype[not_self_loop_idx]]
    print('There are {} self-loops and {} duplicated edges in the removed edges'.format(len(self_loop_edges[0]),
                                                                                        len(duplicate_edges[0])))

with open(args.schema) as json_file:
    schema = json.load(json_file)
nid_ranges = schema['nid']
eid_ranges = schema['eid']
nid_ranges = {key: np.array(nid_ranges[key]).reshape(
    1, 2) for key in nid_ranges}
eid_ranges = {key: np.array(eid_ranges[key]).reshape(
    1, 2) for key in eid_ranges}
id_map = dgl.distributed.id_map.IdMap(nid_ranges)

ntypes = [(key, nid_ranges[key][0, 0]) for key in nid_ranges]
ntypes.sort(key=lambda e: e[1])
ntype_offset_np = np.array([e[1] for e in ntypes])
ntypes = [e[0] for e in ntypes]
ntypes_map = {e: i for i, e in enumerate(ntypes)}
etypes = [(key, eid_ranges[key][0, 0]) for key in eid_ranges]
etypes.sort(key=lambda e: e[1])
etype_offset_np = np.array([e[1] for e in etypes])
etypes = [e[0] for e in etypes]
etypes_map = {e: i for i, e in enumerate(etypes)}


def read_feats(file_name):
    attrs = csv.read_csv(file_name, read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
    num_cols = len(attrs.columns)
    return np.stack([attrs.columns[i].to_numpy() for i in range(num_cols)], 1)


max_nid = np.iinfo(np.int32).max
num_edges = 0
num_nodes = 0
node_map_val = {ntype: [] for ntype in ntypes}
edge_map_val = {etype: [] for etype in etypes}
for part_id in range(num_parts):
    part_dir = output_dir + '/part' + str(part_id)
    os.makedirs(part_dir, exist_ok=True)

    node_file = 'p{:03}-{}_nodes.txt'.format(part_id, graph_name)
    # The format of each line in the node file:
    # <node_id> <node_type> <weight1> ... <orig_type_node_id> <attributes>
    # The node file contains nodes that belong to a partition. It doesn't include HALO nodes.
    orig_type_nid_col = 3 + num_node_weights
    first_attr_col = 4 + num_node_weights

    # Get the first two columns which is the node ID and node type.
    tmp_output = workspace_dir + '/' + node_file + '.tmp'
    os.system('awk \'{print $1, $2, $' + str(orig_type_nid_col) + '}\''
              + ' {} > {}'.format(input_dir + '/' + node_file, tmp_output))
    nodes = csv.read_csv(tmp_output, read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
    nids, ntype_ids, orig_type_nid = nodes.columns[0].to_numpy(), nodes.columns[1].to_numpy(), \
        nodes.columns[2].to_numpy()
    orig_homo_nid = ntype_offset_np[ntype_ids] + orig_type_nid
    assert np.all(nids[1:] - nids[:-1] == 1)
    nid_range = (nids[0], nids[-1])
    num_nodes += len(nodes)

    if node_attr_dtype is not None:
        # Get node attributes
        # Here we just assume all nodes have the same attributes.
        # In practice, this is not the same, in which we need more complex solution to
        # encode and decode node attributes.
        os.system('cut -d\' \' -f {}- {} > {}'.format(first_attr_col,
                                                      input_dir + '/' + node_file,
                                                      tmp_output))
        node_attrs = read_feats(tmp_output)
        node_feats = {}
        # nodes in a partition has been sorted based on node types.
        for ntype_name in nid_ranges:
            ntype_id = ntypes_map[ntype_name]
            type_nids = nids[ntype_ids == ntype_id]
            assert np.all(type_nids == np.arange(
                type_nids[0], type_nids[-1] + 1))
            node_feats[ntype_name +
                       '/feat'] = th.as_tensor(node_attrs[ntype_ids == ntype_id])
        dgl.data.utils.save_tensors(os.path.join(
            part_dir, "node_feat.dgl"), node_feats)

    # Determine the node ID ranges of different node types.
    for ntype_name in nid_ranges:
        ntype_id = ntypes_map[ntype_name]
        type_nids = nids[ntype_ids == ntype_id]
        node_map_val[ntype_name].append(
            [int(type_nids[0]), int(type_nids[-1]) + 1])

    edge_file = 'p{:03}-{}_edges.txt'.format(part_id, graph_name)
    # The format of each line in the edge file:
    # <src_id> <dst_id> <orig_src_id> <orig_dst_id> <orig_edge_id> <edge_type> <attributes>

    tmp_output = workspace_dir + '/' + edge_file + '.tmp'
    os.system('awk \'{print $1, $2, $3, $4, $5, $6}\'' + ' {} > {}'.format(input_dir + '/' + edge_file,
                                                                           tmp_output))
    edges = csv.read_csv(tmp_output, read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
    num_cols = len(edges.columns)
    src_id, dst_id, orig_src_id, orig_dst_id, orig_edge_id, etype_ids = [
        edges.columns[i].to_numpy() for i in range(num_cols)]

    # Let's merge the self-loops and duplicated edges to the partition.
    src_id_list, dst_id_list = [src_id], [dst_id]
    orig_src_id_list, orig_dst_id_list = [orig_src_id], [orig_dst_id]
    orig_edge_id_list, etype_id_list = [orig_edge_id], [etype_ids]
    if self_loop_edges is not None and len(self_loop_edges[0]) > 0:
        uniq_orig_nids, idx = np.unique(orig_dst_id, return_index=True)
        common_nids, common_idx1, common_idx2 = np.intersect1d(
            uniq_orig_nids, self_loop_edges[0], return_indices=True)
        idx = idx[common_idx1]
        # the IDs after ID assignment
        src_id_list.append(dst_id[idx])
        dst_id_list.append(dst_id[idx])
        # homogeneous IDs in the input graph.
        orig_src_id_list.append(self_loop_edges[0][common_idx2])
        orig_dst_id_list.append(self_loop_edges[0][common_idx2])
        # edge IDs and edge type.
        orig_edge_id_list.append(self_loop_edges[2][common_idx2])
        etype_id_list.append(self_loop_edges[3][common_idx2])
        print('Add {} self-loops in partition {}'.format(len(idx), part_id))
    if duplicate_edges is not None and len(duplicate_edges[0]) > 0:
        part_ids = orig_src_id.astype(
            np.int64) * max_nid + orig_dst_id.astype(np.int64)
        uniq_orig_ids, idx = np.unique(part_ids, return_index=True)
        duplicate_ids = duplicate_edges[0].astype(
            np.int64) * max_nid + duplicate_edges[1].astype(np.int64)
        common_nids, common_idx1, common_idx2 = np.intersect1d(
            uniq_orig_ids, duplicate_ids, return_indices=True)
        idx = idx[common_idx1]
        # the IDs after ID assignment
        src_id_list.append(src_id[idx])
        dst_id_list.append(dst_id[idx])
        # homogeneous IDs in the input graph.
        orig_src_id_list.append(duplicate_edges[0][common_idx2])
        orig_dst_id_list.append(duplicate_edges[1][common_idx2])
        # edge IDs and edge type.
        orig_edge_id_list.append(duplicate_edges[2][common_idx2])
        etype_id_list.append(duplicate_edges[3][common_idx2])
        print('Add {} duplicated edges in partition {}'.format(len(idx), part_id))
    src_id = np.concatenate(src_id_list) if len(
        src_id_list) > 1 else src_id_list[0]
    dst_id = np.concatenate(dst_id_list) if len(
        dst_id_list) > 1 else dst_id_list[0]
    orig_src_id = np.concatenate(orig_src_id_list) if len(
        orig_src_id_list) > 1 else orig_src_id_list[0]
    orig_dst_id = np.concatenate(orig_dst_id_list) if len(
        orig_dst_id_list) > 1 else orig_dst_id_list[0]
    orig_edge_id = np.concatenate(orig_edge_id_list) if len(
        orig_edge_id_list) > 1 else orig_edge_id_list[0]
    etype_ids = np.concatenate(etype_id_list) if len(
        etype_id_list) > 1 else etype_id_list[0]
    print('There are {} edges in partition {}'.format(len(src_id), part_id))

    # It's not guaranteed that the edges are sorted based on edge type.
    # Let's sort edges and all attributes on the edges.
    sort_idx = np.argsort(etype_ids)
    src_id, dst_id, orig_src_id, orig_dst_id, orig_edge_id, etype_ids = src_id[sort_idx], dst_id[sort_idx], \
        orig_src_id[sort_idx], orig_dst_id[sort_idx], orig_edge_id[sort_idx], etype_ids[sort_idx]
    assert np.all(np.diff(etype_ids) >= 0)

    if edge_attr_dtype is not None:
        # Get edge attributes
        # Here we just assume all edges have the same attributes.
        # In practice, this is not the same, in which we need more complex solution to
        # encode and decode edge attributes.
        os.system('cut -d\' \' -f 7- {} > {}'.format(input_dir +
                  '/' + edge_file, tmp_output))
        edge_attrs = th.as_tensor(read_feats(tmp_output))[sort_idx]
        edge_feats = {}
        for etype_name in eid_ranges:
            etype_id = etypes_map[etype_name]
            edge_feats[etype_name +
                       '/feat'] = th.as_tensor(edge_attrs[etype_ids == etype_id])
        dgl.data.utils.save_tensors(os.path.join(
            part_dir, "edge_feat.dgl"), edge_feats)

    # Determine the edge ID range of different edge types.
    edge_id_start = num_edges
    for etype_name in eid_ranges:
        etype_id = etypes_map[etype_name]
        edge_map_val[etype_name].append([int(edge_id_start),
                                         int(edge_id_start + np.sum(etype_ids == etype_id))])
        edge_id_start += np.sum(etype_ids == etype_id)

    # Here we want to compute the unique IDs in the edge list.
    # It is possible that a node that belongs to the partition but it doesn't appear
    # in the edge list. That is, the node is assigned to this partition, but its neighbor
    # belongs to another partition so that the edge is assigned to another partition.
    # This happens in a directed graph.
    # To avoid this kind of nodes being removed from the graph, we add node IDs that
    # belong to this partition.
    ids = np.concatenate(
        [src_id, dst_id, np.arange(nid_range[0], nid_range[1] + 1)])
    uniq_ids, idx, inverse_idx = np.unique(
        ids, return_index=True, return_inverse=True)
    assert len(uniq_ids) == len(idx)
    # We get the edge list with their node IDs mapped to a contiguous ID range.
    local_src_id, local_dst_id = np.split(inverse_idx[:len(src_id) * 2], 2)
    compact_g = dgl.graph((local_src_id, local_dst_id))
    compact_g.edata['orig_id'] = th.as_tensor(orig_edge_id)
    compact_g.edata[dgl.ETYPE] = th.as_tensor(etype_ids)
    compact_g.edata['inner_edge'] = th.ones(
        compact_g.number_of_edges(), dtype=th.bool)

    # The original IDs are homogeneous IDs.
    # Similarly, we need to add the original homogeneous node IDs
    orig_ids = np.concatenate([orig_src_id, orig_dst_id, orig_homo_nid])
    orig_homo_ids = orig_ids[idx]
    ntype, per_type_ids = id_map(orig_homo_ids)
    compact_g.ndata['orig_id'] = th.as_tensor(per_type_ids)
    compact_g.ndata[dgl.NTYPE] = th.as_tensor(ntype)
    compact_g.ndata[dgl.NID] = th.as_tensor(uniq_ids)
    compact_g.ndata['inner_node'] = th.as_tensor(np.logical_and(
        uniq_ids >= nid_range[0], uniq_ids <= nid_range[1]))
    local_nids = compact_g.ndata[dgl.NID][compact_g.ndata['inner_node'].bool()]
    assert np.all((local_nids == th.arange(
        local_nids[0], local_nids[-1] + 1)).numpy())
    print('|V|={}'.format(compact_g.number_of_nodes()))
    print('|E|={}'.format(compact_g.number_of_edges()))

    # We need to reshuffle nodes in a partition so that all local nodes are labelled starting from 0.
    reshuffle_nodes = th.arange(compact_g.number_of_nodes())
    reshuffle_nodes = th.cat([reshuffle_nodes[compact_g.ndata['inner_node'].bool()],
                              reshuffle_nodes[compact_g.ndata['inner_node'] == 0]])
    compact_g1 = dgl.node_subgraph(compact_g, reshuffle_nodes)
    compact_g1.ndata['orig_id'] = compact_g.ndata['orig_id'][reshuffle_nodes]
    compact_g1.ndata[dgl.NTYPE] = compact_g.ndata[dgl.NTYPE][reshuffle_nodes]
    compact_g1.ndata[dgl.NID] = compact_g.ndata[dgl.NID][reshuffle_nodes]
    compact_g1.ndata['inner_node'] = compact_g.ndata['inner_node'][reshuffle_nodes]
    compact_g1.edata['orig_id'] = compact_g.edata['orig_id'][compact_g1.edata[dgl.EID]]
    compact_g1.edata[dgl.ETYPE] = compact_g.edata[dgl.ETYPE][compact_g1.edata[dgl.EID]]
    compact_g1.edata['inner_edge'] = compact_g.edata['inner_edge'][compact_g1.edata[dgl.EID]]

    # reshuffle edges on ETYPE as node_subgraph relabels edges
    idx = th.argsort(compact_g1.edata[dgl.ETYPE])
    u, v = compact_g1.edges()
    u = u[idx]
    v = v[idx]
    compact_g2 = dgl.graph((u, v))
    compact_g2.ndata['orig_id'] = compact_g1.ndata['orig_id']
    compact_g2.ndata[dgl.NTYPE] = compact_g1.ndata[dgl.NTYPE]
    compact_g2.ndata[dgl.NID] = compact_g1.ndata[dgl.NID]
    compact_g2.ndata['inner_node'] = compact_g1.ndata['inner_node']
    compact_g2.edata['orig_id'] = compact_g1.edata['orig_id'][idx]
    compact_g2.edata[dgl.ETYPE] = compact_g1.edata[dgl.ETYPE][idx]
    compact_g2.edata['inner_edge'] = compact_g1.edata['inner_edge'][idx]
    compact_g2.edata[dgl.EID] = th.arange(
        num_edges, num_edges + compact_g2.number_of_edges())
    num_edges += compact_g2.number_of_edges()

    dgl.save_graphs(part_dir + '/graph.dgl', [compact_g2])

part_metadata = {'graph_name': graph_name,
                 'num_nodes': num_nodes,
                 'num_edges': num_edges,
                 'part_method': 'metis',
                 'num_parts': num_parts,
                 'halo_hops': 1,
                 'node_map': node_map_val,
                 'edge_map': edge_map_val,
                 'ntypes': ntypes_map,
                 'etypes': etypes_map}

for part_id in range(num_parts):
    part_dir = 'part' + str(part_id)
    node_feat_file = os.path.join(part_dir, "node_feat.dgl")
    edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
    part_graph_file = os.path.join(part_dir, "graph.dgl")
    part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                'edge_feats': edge_feat_file,
                                                'part_graph': part_graph_file}
with open('{}/{}.json'.format(output_dir, graph_name), 'w') as outfile:
    json.dump(part_metadata, outfile, sort_keys=True, indent=4)
