from idpanel.training.vectorization import vectorize
from idpanel.training.features import load_raw_features
from idpanel.labels import load_labels
from idpanel.blacklist import labels_to_ignore, feature_blacklist
import json
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

USE_MULTIPROCESS = True  
# Set to false in case of getting errors in multiprocessing synchronization

_raw_features = None
_sites = None


def preload_process(sites):
    global _raw_features, _sites
    _raw_features = load_raw_features()
    _sites = sites


def compute_vectors(site):
    global _raw_features, _sites
    return list(vectorize(_raw_features, _sites[site])), site


if __name__ == "__main__":
    print("Loading prevectors")
    data_points = []
    with open("prevectors.json", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            line = json.loads(line)
            data_points.append(line)

    label_indeces = load_labels()
    raw_features = load_raw_features()
    print("Loaded {0} features".format(len(raw_features)))

    print("Grouping prevectors by base_url")
    sites = {}
    site_labels = {}
    for dp in data_points:
        if any(bl in dp['offset'] for bl in feature_blacklist):
            continue
        if dp['base_url'] not in sites:
            sites[dp['base_url']] = {}
            site_labels[dp['base_url']] = dp['label']

        sites[dp['base_url']][dp['offset']] = {"code": dp['code'], "content_ssdeep": dp['content_ssdeep']}

    print("Vectorizing {0} base urls".format(len(sites)))
    labels = []
    names = []
    vectors = []
    if USE_MULTIPROCESS:
        pool = Pool(processes=cpu_count(), initializer=preload_process, initargs=(sites,))
        siteiterator = pool.imap_unordered(compute_vectors, sites.keys())
    else:
        preload_process(sites)
        siteiterator = map(compute_vectors, sites.keys())
    for vector, site in siteiterator:
        if site_labels[site] in labels_to_ignore:
            continue
        vectors.append(vector)
        labels.append(site_labels[site])
        names.append(site)
        print("Vector for {0} completed".format(site))

    with open("raw_feature_vectors.json", "w") as f:
        json.dump({"labels": labels, "names": names, "vectors": vectors}, f)
