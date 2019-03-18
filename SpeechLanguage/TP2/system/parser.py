from cyk import CYK, parsing
from oov import OOV
from utils import *
from pathlib import Path 


if __name__ == "__main__":

    parser = ArgumentParser(description="Parse a sentence using CYK algorithm")
    parser.add_argument("--database", "-d", 
            default=Path("system/data/sequoia-corpus+fct.mrg_strict"),
            type=Path, help="Path to SEQUOIA database")
    parser.add_argument("--sentence", "-s", 
           type=str, help="Sentence to parse")
 

    args = parser.parse_args()

    db = Dataset(args.database)
    pcfg = db.get_pcfg(TRAINING)
    train_sentences = db.get_sentences(TRAINING)
    cyk = CYK(pcfg)
    oov = OOV(train_sentences)
    

    print(parsing(cyk, oov.correct_word, [args.sentence]))
