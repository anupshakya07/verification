from hmln import CitationNetworkHMLN
import utils


def main():
    num_nodes = 500
    num_clusters = 50
    num_iters = 30
    dataset = "cora"
    spec_model = "gcn"

    spec_embedding_filepath = f"embeddings/{dataset}/{spec_model}_specification_embeddings.pkl"
    spec_pred_filepath = f"embeddings/{dataset}/{spec_model}_specification_pred_prod.pkl"
    spec_embeddings = utils.load_file(spec_embedding_filepath)
    spec_distance_matrix = utils.compute_distance_matrix(spec_embeddings, num_nodes)
    spec_pred_probabilities = utils.load_file(spec_pred_filepath)

    citation_hmln = CitationNetworkHMLN(dataset="cora", num_nodes=num_nodes, num_clusters=num_clusters,
                                        spec_distance_matrix=spec_distance_matrix)

    return


if __name__ == "__main__":
    main()
