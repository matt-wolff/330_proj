import pcmap
import sys
import pdb

# (ChainID,ResID) format
def procNode(theirForm):
    return (theirForm["chainID"], theirForm["resID"])

def order(pair):
    if pair[0][0] != pair[1][0]:
        return pair[0][0] < pair[1][0]
    return pair[0][1] < pair[1][1]


def buildContactGraph(PDBcode):
    cmap = pcmap.contactMap("data/UP000005640_9606_HUMAN_v4/AF-"+PDBcode+"-F1-model_v4.pdb", dist=8.0)
    #pdb.set_trace()
    # TODO distance parameter

    contacts = [] # Do note this has contacts in either direction and including backbone
    for nodeinf in cmap['data']:
        origin = procNode(nodeinf['root'])
        for dest in nodeinf['partners']:
            con =(origin, procNode(dest))
            if order(con):
                contacts.append(con)

    return contacts

if "__main__" == __name__:
    print(buildContactGraph(sys.argv[1]))
