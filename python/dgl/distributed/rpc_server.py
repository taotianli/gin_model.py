"""Functions used by server."""

import time
import os
from ..base import DGLError
from . import rpc
from .constants import MAX_QUEUE_SIZE, SERVER_EXIT, SERVER_KEEP_ALIVE

def start_server(server_id, ip_config, num_servers, num_clients, server_state, \
    max_queue_size=MAX_QUEUE_SIZE, net_type='tensorpipe'):
    """Start DGL server, which will be shared with all the rpc services.

    This is a blocking function -- it returns only when the server shutdown.

    Parameters
    ----------
    server_id : int
        Current server ID (starts from 0).
    ip_config : str
        Path of IP configuration file.
    num_servers : int
        Server count on each machine.
    num_clients : int
        Total number of clients that will be connected to the server.
        Note that, we do not support dynamic connection for now. It means
        that when all the clients connect to server, no client will can be added
        to the cluster.
    server_state : ServerSate object
        Store in main data used by server.
    max_queue_size : int
        Maximal size (bytes) of server queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound because DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str
        Networking type. Current options are: ``'socket'`` or ``'tensorpipe'``.
    """
    assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
    assert num_servers > 0, 'num_servers (%d) must be a positive number.' % num_servers
    assert num_clients >= 0, 'num_client (%d) cannot be a negative number.' % num_clients
    assert max_queue_size > 0, 'queue_size (%d) cannot be a negative number.' % max_queue_size
    assert net_type in ('socket', 'tensorpipe'), \
        'net_type (%s) can only be \'socket\' or \'tensorpipe\'' % net_type
    if server_state.keep_alive:
        assert net_type == 'tensorpipe', \
            "net_type can only be 'tensorpipe' if 'keep_alive' is enabled."
        print("As configured, this server will keep alive for multiple"
              " client groups until force shutdown request is received."
              " [WARNING] This feature is experimental and not fully tested.")
    # Register signal handler.
    rpc.register_sig_handler()
    # Register some basic services
    rpc.register_service(rpc.CLIENT_REGISTER,
                         rpc.ClientRegisterRequest,
                         rpc.ClientRegisterResponse)
    rpc.register_service(rpc.SHUT_DOWN_SERVER,
                         rpc.ShutDownRequest,
                         None)
    rpc.register_service(rpc.GET_NUM_CLIENT,
                         rpc.GetNumberClientsRequest,
                         rpc.GetNumberClientsResponse)
    rpc.register_service(rpc.CLIENT_BARRIER,
                         rpc.ClientBarrierRequest,
                         rpc.ClientBarrierResponse)
    rpc.set_rank(server_id)
    server_namebook = rpc.read_ip_config(ip_config, num_servers)
    machine_id = server_namebook[server_id][0]
    rpc.set_machine_id(machine_id)
    ip_addr = server_namebook[server_id][1]
    port = server_namebook[server_id][2]
    rpc.create_sender(max_queue_size, net_type)
    rpc.create_receiver(max_queue_size, net_type)
    # wait all the senders connect to server.
    # Once all the senders connect to server, server will not
    # accept new sender's connection
    print(
        "Server is waiting for connections on [{}:{}]...".format(ip_addr, port))
    rpc.wait_for_senders(ip_addr, port, num_clients,
                         blocking=net_type == 'socket')
    rpc.set_num_client(num_clients)
    recv_clients = {}
    while True:
        # go through if any client group is ready for connection
        for group_id in list(recv_clients.keys()):
            ips = recv_clients[group_id]
            if len(ips) < rpc.get_num_client():
                continue

            del recv_clients[group_id]
            # a new client group is ready
            ips.sort()
            client_namebook = dict(enumerate(ips))
            time.sleep(3) # wait for clients' receivers ready
            max_try_times = int(os.environ.get('DGL_DIST_MAX_TRY_TIMES', 120))
            for client_id, addr in client_namebook.items():
                client_ip, client_port = addr.split(':')
                try_times = 0
                while not rpc.connect_receiver(client_ip, client_port, client_id, group_id):
                    try_times += 1
                    if try_times % 200 == 0:
                        print("Server~{} is trying to connect client receiver: {}:{}".format(
                            server_id, client_ip, client_port))
                    if try_times >= max_try_times:
                        raise rpc.DistConnectError(max_try_times, client_ip, client_port)
                    time.sleep(1)
            if not rpc.connect_receiver_finalize(max_try_times):
                raise rpc.DistConnectError(max_try_times)
            if rpc.get_rank() == 0:  # server_0 send all the IDs
                for client_id, _ in client_namebook.items():
                    register_res = rpc.ClientRegisterResponse(client_id)
                    rpc.send_response(client_id, register_res, group_id)
        # receive incomming client requests
        timeout = 60 * 1000  # in milliseconds
        req, client_id, group_id = rpc.recv_request(timeout)
        if req is None:
            continue
        if isinstance(req, rpc.ClientRegisterRequest):
            if group_id not in recv_clients:
                recv_clients[group_id] = []
            recv_clients[group_id].append(req.ip_addr)
            continue

        res = req.process_request(server_state)
        if res is not None:
            if isinstance(res, list):
                for response in res:
                    target_id, res_data = response
                    rpc.send_response(target_id, res_data, group_id)
            elif isinstance(res, str):
                if res == SERVER_EXIT:
                    print("Server is exiting...")
                    return
                elif res == SERVER_KEEP_ALIVE:
                    print("Server keeps alive while client group~{} is exiting...".format(group_id))
                else:
                    raise DGLError("Unexpected response: {}".format(res))
            else:
                rpc.send_response(client_id, res, group_id)
