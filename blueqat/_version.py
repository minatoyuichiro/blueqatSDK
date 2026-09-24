# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The version of blueqat, and which build of it is installed.

Two numbers, because one is not enough. `__version__` says which release this
is. `installed_revision()` says which *build* -- and between releases that is
the question people actually have.

It is the question because installs are pinned by commit: ``pip install
git+https://github.com/blueqat/blueqatSDK@<sha>``. Several such builds share a
version, and a run reporting "2.1.3" said nothing about which. That cost real
time: a session reported a bug against "2.1.3" that another could not
reproduce on "2.1.3", and settling it took reading `direct_url.json` by hand
across three machines. `version_info()` answers it in one call.
"""

from typing import Dict, Optional

#: The release. Bumped when the contents change, which is the part that had
#: been missed: 2.1.3 was cut on 2026-07-15 and then carried 17,666 added
#: lines and nine new modules without moving.
__version__: str = "2.2.0"


def installed_revision() -> Optional[str]:
    """The commit this build came from, if that is recorded, else None.

    Decided from the file that is actually imported, not from whichever
    distribution metadata happens to be found first. That distinction is not
    hypothetical: a source checkout carries a `blueqat.egg-info`, which
    shadows an installed `dist-info` whenever the working directory is the
    repository -- so the same call answered differently depending on where it
    was run from, and could have reported an installed commit while running
    checkout code. Reporting the wrong revision confidently is worse than
    reporting none.

    A git checkout answers with its commit, plus ``-dirty`` when the working
    tree differs, because "the commit" is not the whole truth while somebody
    is editing. An install pinned with ``pip install git+...@<sha>`` answers
    with the commit pip recorded. A release from PyPI records no commit and
    answers None, which is an answer rather than a failure.
    """
    return _revision_from_source() or _revision_from_metadata()


def _distribution_containing_this_file():
    """The installed distribution this module belongs to, or None.

    `distribution('blueqat')` is not enough: it returns the first match, which
    may describe a different copy from the one that got imported.
    """
    from importlib.metadata import distributions
    from pathlib import Path
    here = Path(__file__).resolve()
    for dist in distributions():
        if (dist.metadata.get('Name') or '').lower() != 'blueqat':
            continue
        try:
            root = Path(dist.locate_file('')).resolve()
        except Exception:
            continue
        if root in here.parents:
            return dist
    return None


def _revision_from_metadata() -> Optional[str]:
    try:
        import json
        dist = _distribution_containing_this_file()
        if dist is None:
            return None
        text = dist.read_text('direct_url.json')
        if not text:
            return None
        info = json.loads(text).get('vcs_info') or {}
        return info.get('commit_id') or None
    except Exception:
        return None


def _revision_from_source() -> Optional[str]:
    """The git revision of the checkout this file sits in, if it is one."""
    try:
        import subprocess
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        if not (root / '.git').exists():
            return None
        def git(*args: str) -> str:
            return subprocess.run(('git', '-C', str(root)) + args,
                                  capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        commit = git('rev-parse', 'HEAD')
        if not commit:
            return None
        return commit + ('-dirty' if git('status', '--porcelain') else '')
    except Exception:
        return None


def version_info() -> Dict[str, Optional[str]]:
    """Everything needed to say which blueqat this is, in one call.

    ``{'version', 'revision', 'source', 'path'}``. `source` is where the
    revision came from -- ``'install'``, ``'checkout'`` or ``None`` -- because
    a commit read out of a git checkout means something different from one
    recorded by pip, and a bug report that does not distinguish them sends
    people looking in the wrong place.
    """
    from_source = _revision_from_source()
    revision = from_source or _revision_from_metadata()
    return {
        'version': __version__,
        'revision': revision,
        'source': ('checkout' if from_source
                   else ('install' if revision else None)),
        'path': str(__import__('pathlib').Path(__file__).resolve().parent),
    }
