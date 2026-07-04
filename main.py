"""CLI: python main.py {train|eval} ... (skeleton — experiments gated per docs/PROPOSAL.md)."""
import sys, runpy, os
_H = os.path.dirname(os.path.abspath(__file__))
cmds = {'train': 'train.py', 'eval': 'eval.py'}
if len(sys.argv) < 2 or sys.argv[1] not in cmds:
    print(__doc__); sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
sys.argv = [os.path.join(_H, cmds[sys.argv[1]])] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
