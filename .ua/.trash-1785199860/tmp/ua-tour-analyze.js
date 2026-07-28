const fs = require('fs');

function main() {
  const inPath = process.argv[2], outPath = process.argv[3];
  const data = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  const nodes = data.nodes || [], edges = data.edges || [], layers = data.layers || [];

  const FILE_TYPES = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
  const fileNodes = nodes.filter(n => FILE_TYPES.has(n.type));
  const fileIds = new Set(fileNodes.map(n => n.id));
  const byId = {}; nodes.forEach(n => byId[n.id] = n);

  const fanIn = {}, fanOut = {};
  fileNodes.forEach(n => { fanIn[n.id] = 0; fanOut[n.id] = 0; });
  for (const e of edges) {
    if (!fileIds.has(e.source) || !fileIds.has(e.target)) continue;
    fanOut[e.source]++; fanIn[e.target]++;
  }
  const rank = (m, key) => Object.entries(m).map(([id, v]) => ({ id, [key]: v, name: byId[id].name }))
    .sort((a, b) => b[key] - a[key]).slice(0, 20);

  // entry points
  const ENTRY = new Set(['index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js','mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py','Application.java','Main.java','Program.cs','config.ru','index.php','App.swift','Application.kt','main.cpp','main.c']);
  const foVals = Object.values(fanOut).sort((a, b) => b - a);
  const fiVals = Object.values(fanIn).sort((a, b) => a - b);
  const foTop10 = foVals[Math.floor(foVals.length * 0.1)] ?? 0;
  const fiBot25 = fiVals[Math.floor(fiVals.length * 0.25)] ?? 0;
  const cands = fileNodes.map(n => {
    let s = 0;
    const fp = n.filePath || '';
    const depth = fp.split('/').length;
    if (n.type === 'document') {
      if (/^README\.md$/i.test(fp)) s += 5;
      else if (depth === 1 && /\.md$/i.test(fp)) s += 2;
    } else {
      if (ENTRY.has(n.name)) s += 3;
      if (depth <= 2) s += 1;
      if (fanOut[n.id] >= foTop10) s += 1;
      if (fanIn[n.id] <= fiBot25) s += 1;
    }
    return { id: n.id, score: s, name: n.name, type: n.type, summary: n.summary };
  }).filter(c => c.score > 0).sort((a, b) => b.score - a.score);

  // BFS from top code entry point
  const codeEntry = cands.find(c => c.type !== 'document');
  const adj = {};
  for (const e of edges) {
    if (!fileIds.has(e.source) || !fileIds.has(e.target)) continue;
    if (e.type !== 'imports' && e.type !== 'calls' && e.type !== 'depends_on') continue;
    (adj[e.source] = adj[e.source] || []).push(e.target);
  }
  // prefer the highest-fan-out code file as BFS root (real runtime entry)
  const foRanked = rank(fanOut, 'fanOut').filter(x => byId[x.id].type === 'file');
  const start = (foRanked[0] && foRanked[0].id) || (codeEntry ? codeEntry.id : fileNodes[0].id);
  const order = [], depthMap = {}; const q = [start]; depthMap[start] = 0;
  while (q.length) {
    const cur = q.shift(); order.push(cur);
    for (const nx of (adj[cur] || [])) if (!(nx in depthMap)) { depthMap[nx] = depthMap[cur] + 1; q.push(nx); }
  }
  const byDepth = {};
  for (const [id, d] of Object.entries(depthMap)) (byDepth[d] = byDepth[d] || []).push(id);

  const pick = t => fileNodes.filter(n => t.includes(n.type)).map(n => ({ id: n.id, name: n.name, type: n.type, summary: n.summary }));
  const nonCodeFiles = {
    documentation: pick(['document']),
    infrastructure: pick(['service', 'pipeline', 'resource']),
    data: pick(['table', 'schema', 'endpoint']),
    config: pick(['config'])
  };

  // clusters
  const pairKey = (a, b) => a < b ? a + '||' + b : b + '||' + a;
  const pc = {};
  for (const e of edges) {
    if (!fileIds.has(e.source) || !fileIds.has(e.target) || e.source === e.target) continue;
    pc[pairKey(e.source, e.target)] = (pc[pairKey(e.source, e.target)] || 0) + 1;
  }
  const seeds = Object.entries(pc).filter(([, c]) => c >= 2).sort((a, b) => b[1] - a[1]);
  const clusters = []; const used = new Set();
  for (const [k, c] of seeds) {
    const [a, b] = k.split('||');
    if (used.has(a) && used.has(b)) continue;
    const cl = new Set([a, b]);
    for (const n of fileNodes) {
      if (cl.size >= 5 || cl.has(n.id)) continue;
      let links = 0;
      for (const m of cl) if (pc[pairKey(n.id, m)]) links++;
      if (links >= 2) cl.add(n.id);
    }
    [...cl].forEach(x => used.add(x));
    clusters.push({ nodes: [...cl], edgeCount: c });
    if (clusters.length >= 10) break;
  }

  const nodeSummaryIndex = {};
  nodes.forEach(n => nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary, filePath: n.filePath, layerHint: undefined });

  const res = {
    scriptCompleted: true,
    entryPointCandidates: cands.slice(0, 8),
    fanInRanking: rank(fanIn, 'fanIn'),
    fanOutRanking: rank(fanOut, 'fanOut'),
    bfsTraversal: { startNode: start, order, depthMap, byDepth },
    nonCodeFiles, clusters,
    layers: { count: layers.length, list: layers },
    nodeSummaryIndex,
    totalNodes: nodes.length, totalEdges: edges.length,
    fileNodeIds: [...fileIds]
  };
  fs.writeFileSync(outPath, JSON.stringify(res, null, 1));
}
try { main(); } catch (e) { console.error(e); process.exit(1); }
