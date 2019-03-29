# -*- mode: python -*-

block_cipher = None


a = Analysis(['bbsite/manage.py'],
             pathex=['/Users/tan/Downloads/TSP'],
             binaries=None,
             datas=[('bbsite/bbn/migrations', 'bbn/migrations'), ('bbsite/static/some_static', 'static'),
                    ('bbsite/templates/registration', 'registration/migrations')],
             hiddenimports=['django.template.defaulttags', 'django.template.loader_tags', 
                            '_sysconfigdata', 'django.contrib.sessions', 'django.contrib.admin',
                            'django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions',
                            'django.contrib.messages', 'django.contrib.staticfiles', 'django.contrib.sites',
                            'bbn', 'apps', 'whitenoise.storage', 'django.core.context_processors',
                            'django.core.mail.backends.smtp', 'django.views.defaults', 'django.templatetags.i18n',
                            'django.templatetags.tz', 'django.templatetags.l10n', 'django.templatetags.cache',
                            'django.templatetags.future', 'crispy_forms.templatetags.crispy_forms_tags',
                            'crispy_forms.templatetags.crispy_forms_utils', 'crispy_forms.templatetags.crispy_forms_field',
                            'crispy_forms.templatetags.crispy_forms_filters', 'django.contrib.admin.templatetags.admin_static',
                            'django.contrib.admin.templatetags.admin_list', 'django.contrib.admin.templatetags.admin_modify',
                            'django.contrib.admin.templatetags.admin_urls', 'django.contrib.admin.templatetags.log'],
             hookspath=[],
             runtime_hooks=['bbsite/bbn/apps.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='manage',
          debug=False,
          strip=False,
          upx=True,
          console=False )
app = BUNDLE(exe,
             name='manage.app',
             icon=None,
             bundle_identifier=None)
