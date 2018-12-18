from __future__ import print_function # for compatibility with Python 2
from subprocess import call, run, PIPE, check_output, CalledProcessError
import os
from os.path import join
import platform
import sys
import shutil

if 'cwd' in locals():
    os.chdir(cwd)
else:
    cwd = os.getcwd()


# define python installer
if 'Anaconda' in sys.version:
    pyinstaller = 'conda'
elif run(['which', 'pip3'], stdout=PIPE).stdout.decode('utf-8')!='':
    pyinstaller = 'pip3'
else:
    pyinstaller = 'pip'
    
vlfeat_ver = 'vlfeat-0.9.20'

#f = open('log.txt', 'w') # create file for logging the installation output for debugging if needed
try:
    # install cython
    print('installing cython...',end=' ')
    call(pyinstaller+' install cython', shell=True)
    print('done!')
    if not os.path.isfile(vlfeat_ver+'-bin.tar.gz'):
        # download vlfeat binaries
        print('downloading vlfeat binaries...',end=' ')
        check_output('wget http://www.vlfeat.org/download/'+vlfeat_ver+'-bin.tar.gz', shell=True)
        print('done!')
    if not os.path.isdir(vlfeat_ver):
        print('uncompressing vlfeat binaries...',end=' ')
        check_output('tar -xvzf '+vlfeat_ver+'-bin.tar.gz', shell=True)
        print('done!')
    # clone cyvlfeat
    if not os.path.isdir('cyvlfeat_git'):
        print('cloning pyvlfeat repo to cyvlfeat_git folder...',end=' ')
        check_output('git clone https://github.com/ignacio-rocco/cyvlfeat.git cyvlfeat_git', shell=True)
        print('done!')
    # build cyvlfeat
    print('building cyvlfeat...',end=' ')
    os.chdir('cyvlfeat_git/')
    #!python setup.py build
    check_output('VLFEAT_DIR="$(dirname `pwd`)/'+vlfeat_ver+'/" python setup.py build', shell=True)
    print('done!')
    print('installing cyvlfeat...',end=' ')
    check_output("python setup.py install --single-version-externally-managed --record ../cyvlfeat_files.txt", shell=True)
    os.chdir('../')
    print('done! NOTE: to uninstall run: cat cyvlfeat_files.txt | xargs rm -rf')
    print('copying vlfeat dynamic libary to a reachable location...',end=' ')
    if platform.system()=='Linux':
        vl_lib_file = join(cwd,vlfeat_ver,'bin','glnxa64','libvl.so')
        try:    
            shutil.copy(vl_lib_file,'/usr/lib/')
        except:
            shutil.copy(vl_lib_file,'.')
            print('WARNING: The dynamic lib has been copied to your current directory.')
            print('To make it accessible please run:')
            print('       export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH')
            print('and restart the notebook server')
    elif platform.system()=='Windows':
        vl_lib_file = join(cwd,vlfeat_ver,'bin','win64','vl.dll')
        shutil.copy(vl_lib_file,'.')
    elif platform.system()=='Darwin':
        vl_lib_file = join(cwd,vlfeat_ver,'bin','maci64','libvl.dylib')
        try:    
            shutil.copy(vl_lib_file,'/usr/local/lib/')
        except:
            shutil.copy(vl_lib_file,'.')
            print('WARNING: The dynamic lib has been copied to your current directory.')
            print('To make it accessible please run:')
            print('       export ANACONDA_HOME=$HOME/anaconda')
            print('       export DYLD_FALLBACK_LIBRARY_PATH=$ANACONDA_HOME/lib:/usr/local/lib:/usr/lib:`pwd`')
            print('and restart the notebook server')        
    print('done!')
    print('Installation of cyvlfeat is finished')
except CalledProcessError as e:
    print(str(error, 'utf-8'))
    print('failed!')

