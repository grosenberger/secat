import click
from preprocess import uniprot, net

@click.group(chain=True)
@click.version_option()
def cli():
    """
    SECAT: Size-Exclusion Chromatography Algorithmic Toolkit.

    Visit https://github.com/grosenberger/secat for usage instructions and help.
    """

# SECAT import data
@cli.command()
@click.argument('infiles', nargs=-1, type=click.Path(exists=True))
@click.option('--out', 'outfile', required=True, type=click.Path(exists=False), help='Reference binary protein-protein interaction file.')
# # Reference files
@click.option('--sec', 'secfile', type=click.Path(exists=False), help='The input SEC calibration file.')
@click.option('--net', 'netfile', required=True, type=click.Path(exists=True), help='Reference binary protein-protein interaction file in STRING-DB  or HUPO-PSI MITAB (2.5-2.7) format.')
@click.option('--uniprot', 'uniprotfile', type=click.Path(exists=True), help='Reference molecular weights file in UniProt XML format.')
def preprocess(infiles, outfile, secfile, netfile, uniprotfile):
    """
    Import and preprocess SEC data.
    """
    up = uniprot(uniprotfile)

    print net(netfile, up).df


