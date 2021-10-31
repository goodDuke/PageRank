import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Calculate the probability of choosing a random page from the corpus
    random_probability = (1 - damping_factor) / len(corpus)
    # Calculate the probability of choosing a linked page
    pages_linked_probability = damping_factor / len(corpus[page]) + random_probability
    
    probability_dict = {}
    # Add the pages and their probabilities to the dictionary
    for p in corpus[page]:
        probability_dict.update({p: pages_linked_probability})
    
    for p in corpus:
        if p not in probability_dict:
            probability_dict.update({p: random_probability})
    
    #print(probability_dict)
    return probability_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    sample_page = random.choice(list(corpus))

    # Dictionary where keys correspond to pages and the values correspond 
    # to the number of times a page was choosen as sample
    sample_dict = {sample_page: 1}
    for sample in range(1, n):    
        page_list = []
        probability_list = []
        transition_dict = transition_model(corpus, sample_page, damping_factor)

        # Create the two lists needed for the random.choises() method
        for page in transition_dict:
            page_list.append(page)
            probability_list.append(transition_dict[page])
        
        sample_page = random.choices(page_list, probability_list)[0]
        if sample_page not in sample_dict:
            sample_dict.update({sample_page: 1})
        else:
            sample_dict[sample_page] += 1
    
    pagerank_dict = {}
    for page in sample_dict:
        pagerank_dict.update({page: sample_dict[page] / n})
    
    return pagerank_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
