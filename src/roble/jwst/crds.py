'''Control CRDS files.'''

from __future__ import annotations
import os
from logging import getLogger
from importlib import resources
from pathlib import Path
import re

from stdatamodels.jwst.datamodels.util import read_metadata
import crds
from crds.client import api

__all__ = ['CRDSContext']

logger = getLogger(__name__)


##
class CRDSContextNoFILE:
    '''Controle CRDS files with no file.'''

    def __init__(self) -> None:
        self._overwrite = False

    def __enter__(self):
        env = os.environ
        if 'CRDS_PATH_SINGLE' in env:
            pass
        elif 'CRDS_PATH' in env:
            pass
        else:
            with resources.path('roble.jwst.lib', 'crds') as pdir:
                pdir.mkdir(exist_ok=True)
                os.environ['CRDS_PATH_SINGLE'] = str(pdir)
            self._overwrite = True

        logger.info(f'JWST CRDS Path: {crds.config.get_crds_path()}')
        return self

    def __exit__(self, type, value, traceback):
        if self._overwrite:
            del os.environ['CRDS_PATH_SINGLE']

    @staticmethod
    def get_pathtofile(filename: str) -> Path:
        '''Get a path to a given reference file.'''
        dummycontext = 'jwst'
        return Path(api.dump_references(dummycontext, [filename])[filename])


class CRDSContext(CRDSContextNoFILE):
    '''Controle CRDS files.'''

    def __init__(self, fname: Path) -> None:
        self.fname = fname
        self.meta = read_metadata(fname)
        super().__init__()

    def get_context(self) -> str:
        '''Get the CRDS context used in the calibration.'''
        return self.meta['meta.ref_file.crds.context_used']

    def get_path(self, key: str) -> Path:
        '''Get a path to a reference file.'''
        context = self.get_context()
        filename = self.get_meta(key)
        return Path(api.dump_references(context, [filename])[filename])

    def get_meta(self, key: str) -> str:
        '''Get a reference file name.

        If not exist, return an empty string.
        '''
        if result := self._get_metareffile(key):
            return crds.config.pop_crds_uri(result)
        return self._find_reffile_from_logs(key)

    def _get_metareffile(self, key: str) -> str:
        '''Get a reference file name already contained in the meta data.

        If not exist, return an empty string.
        '''
        _key = f'meta.ref_file.{key.lower()}.name'
        if _key in self.meta.keys():
            return self.meta[_key]
        return ''

    def _find_reffile_from_logs(self, key: str) -> str:
        '''Search a reference file name from calibration logs.

        If not exist, return an empty string.
        '''
        logs = [log for log in self.meta['cal_logs'].strip('\n') if key.upper() in log]

        if not logs:
            return ''
        elif len(logs) > 1:
            msg = (
                'The calibration log of the data contains more than one sentances'
                f'including the input keyword "{key.upper()}": {logs}.'
            )
            logger.error(msg)
            raise ValueError(msg)

        substr = re.search("'.+'", logs[0])
        assert substr is not None
        return substr.group().strip("'")
