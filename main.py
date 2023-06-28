from hmln import CitationNetworkHMLN
import utils


def main():
    num_nodes = 500
    num_clusters = 50
    num_iters = 30
    dataset = "cora"
    spec_model = "gcn"
    nuv_model = "gat"

    spec_embedding_filepath = f"embeddings/{dataset}/{spec_model}_specification_embeddings.pkl"
    spec_pred_filepath = f"embeddings/{dataset}/{spec_model}_specification_pred_prob.pkl"
    nuv_embedding_fielpath = f"embeddings/{dataset}/{nuv_model}_nuv_embeddings.pkl"
    nuv_pred_filepath = f"embeddings/{dataset}/{nuv_model}_nuv_pred_prob.pkl"

    spec_embeddings = utils.load_file(spec_embedding_filepath)
    spec_distance_matrix = utils.compute_distance_matrix(spec_embeddings, num_nodes)
    spec_pred_probabilities = utils.load_file(spec_pred_filepath)

    nuv_embeddings = utils.load_file(nuv_embedding_fielpath)
    nuv_distance_matrix = utils.compute_distance_matrix(nuv_embeddings, num_nodes)
    nuv_pred_probabilities = utils.load_file(nuv_pred_filepath)

    citation_hmln = CitationNetworkHMLN(dataset="cora", num_nodes=num_nodes, num_clusters=num_clusters,
                                        spec_distance_matrix=spec_distance_matrix, spec_pred_probabilities=spec_pred_probabilities)

    citation_hmln.train(num_iters=num_iters)
    citation_hmln.verify_nuv(nuv_distance_matrix=nuv_distance_matrix, nuv_pred_probabilities=nuv_pred_probabilities)


if __name__ == "__main__":
    main()
