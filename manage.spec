# -*- mode: python -*-

block_cipher = None


a = Analysis(['bbsite/manage.py'],
             pathex=['/Users/tan/Downloads/TSPV2'],
             binaries=[],
             datas=[('/Users/tan/Downloads/TSPV2/bbsite/static', 'static_root')],
             hiddenimports=['django.template.defaulttags', 'django.template.loader_tags', 
                            '_sysconfigdata', 'django.contrib.sessions', 'django.contrib.admin',
                            'django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions',
                            'django.contrib.messages', 'django.contrib.staticfiles', 'django.contrib.sites',
                            'bbn', 'apps', 'whitenoise.storage', 'django.core.context_processors'],
             hookspath=['/Users/tan/Downloads/test/lib/python2.7/site-packages/django/',
                        '/Users/tan/Downloads/TSPV2/bbsite/bbn'],
             runtime_hooks=['/Users/tan/Downloads/TSPV2/bbsite/bbn/apps.py',
                            '/Users/tan/Downloads/test/lib/python2.7/site-packages/django/core/context_processors.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='manage',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False )
app = BUNDLE(exe,
             name='manage.app',
             icon=None,
             bundle_identifier=None)
