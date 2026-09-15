// FlyTalk FlyWire v9 hotfix: iOS may rename *.csv.gz to *.csv.gz.csv.
// Detect gzip from file bytes instead of trusting the filename.
async function flywireFileCompression(file){
  const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  if (head.length >= 2 && head[0] === 0x1f && head[1] === 0x8b) return true;

  // Plain CSV sanity check. This also gives a clearer error for the wrong file.
  const probe = await file.slice(0, 256).text();
  if (/pre_(?:pt_)?root_id/i.test(probe) && /post_(?:pt_)?root_id/i.test(probe)) return false;
  throw new Error('arquivo não parece ser GZIP nem connections_princeton CSV');
}

const flywireFileInput = document.querySelector('#file');
if (flywireFileInput) {
  flywireFileInput.onchange = async e => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      set('#graphState', 'IMPORTANDO');
      set('#importProgress', 'verificando o arquivo…');
      const gz = await flywireFileCompression(file);
      set('#importProgress', gz
        ? 'GZIP detectado pelos bytes (1F 8B) · descompactando…'
        : 'CSV puro detectado · iniciando leitura…');
      status(`arquivo reconhecido como ${gz ? 'GZIP' : 'CSV'} · ${file.name}`);
      await processConnectionStream(file.stream(), gz, 'Codex ' + file.name);
    } catch (err) {
      console.error(err);
      const text = err?.message || String(err);
      msg('sys', 'Erro ao importar: ' + text);
      set('#graphState', 'ERRO');
      set('#importProgress', text);
      status('falha na importação · ' + text);
    }
  };
}
