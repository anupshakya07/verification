from hmln import CitationNetworkHMLN
import utils
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-num_nodes", type=int, default=500, help="Number of nodes to take in training")
    parser.add_argument("-num_clusters", type=int, default=50, help="Number of weight-sharing clusters to form")
    parser.add_argument("-num_iters", type=int, default=30, help="Number of iterations to train the HMLN")
    parser.add_argument("-dataset", type=str, default="cora", choices=["cora", "citeseer", "pubmed"],
                        help="Choose the dataset name")
    parser.add_argument("-spec_model", type=str, default="gcn", choices=["gcn", "gs", "gat"],
                        help="Select the specification DNN.")
    parser.add_argument("-nuv_model", type=str, default="gat",choices=["gcn", "gs", "gat"],
                        help="Select the NUV DNN.")

    opt = parser.parse_args()
    num_nodes = opt.num_nodes
    num_clusters = opt.num_clusters
    num_iters = opt.num_iters
    dataset = opt.dataset
    spec_model = opt.spec_model
    nuv_model = opt.nuv_model

    spec_embedding_filepath = f"embeddings/{dataset}/{spec_model}_specification_embeddings.pkl"
    spec_pred_filepath = f"embeddings/{dataset}/{spec_model}_specification_pred_prob.pkl"
    nuv_embedding_fielpath = f"embeddings/{dataset}/{nuv_model}_nuv_embeddings.pkl"
    nuv_pred_filepath = f"embeddings/{dataset}/{nuv_model}_nuv_pred_prob.pkl"

    spec_embeddings = utils.load_file(spec_embedding_filepath)
    spec_distance_matrix = utils.compute_distance_matrix(spec_embeddings.detach().cpu().numpy(), num_nodes)
    spec_pred_probabilities = utils.load_file(spec_pred_filepath)

    nuv_embeddings = utils.load_file(nuv_embedding_fielpath)
    nuv_distance_matrix = utils.compute_distance_matrix(nuv_embeddings.detach().cpu().numpy(), num_nodes)
    nuv_pred_probabilities = utils.load_file(nuv_pred_filepath)

    citation_hmln = CitationNetworkHMLN(dataset="cora", num_nodes=num_nodes, num_clusters=num_clusters,
                                        spec_distance_matrix=spec_distance_matrix, spec_pred_probabilities=spec_pred_probabilities)

    citation_hmln.train(num_iters=num_iters)
    citation_hmln.verify_nuv(nuv_distance_matrix=nuv_distance_matrix, nuv_pred_probabilities=nuv_pred_probabilities)


if __name__ == "__main__":
    main()
