#!/usr/bin/env node
const fs = require('fs');

function main() {
  const inPath = process.argv[2], outPath = process.argv[3];
  const data = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  const fileNodes = data.fileNodes || [];
  const importEdges = data.importEdges || [];
  const allEdges = data.allEdges || [];

  const byId = new Map(fileNodes.map(n => [n.id, n]));
  const paths = fileNodes.map(n => n.filePath || '');

  // --- common prefix (directory-segment-wise) ---
  const segLists = paths.map(p => p.split('/'));
  let prefix = [];
  if (segLists.length) {
    const first = segLists[0];
    for (let i = 0; i < first.length - 1; i++) {
      const s = first[i];
      if (segLists.every(sl => sl.length > i + 1 && sl[i] === s)) prefix.push(s); else break;
    }
  }
  // don't strip prefix if it swallows everything into one group
  // descend past any single directory that holds the overwhelming majority of files
  const descend = [];
  for (let depth = 0; depth < 4; depth++) {
    const counts = {};
    for (const p of paths) {
      const segs = p.split('/');
      if (segs.length <= descend.length + 1) continue;
      if (!descend.every((s, i) => segs[i] === s)) continue;
      const g = segs[descend.length];
      counts[g] = (counts[g] || 0) + 1;
    }
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    if (entries.length && entries[0][1] / paths.length > 0.6) descend.push(entries[0][0]);
    else break;
  }
  prefix = descend;

  const groupOf = (p) => {
    let segs = p.split('/');
    if (prefix.length && segs.length > prefix.length && prefix.every((s, i) => segs[i] === s)) {
      segs = segs.slice(prefix.length);
      return segs.length > 1 ? segs[0] : `(${prefix.join('/')}-root)`;
    }
    return segs.length > 1 ? segs[0] : '(root)';
  };

  // A. directory groups
  const directoryGroups = {};
  for (const n of fileNodes) {
    const g = groupOf(n.filePath || '');
    (directoryGroups[g] = directoryGroups[g] || []).push(n.id);
  }
  // if only one group and it has subdirs, descend one level
  const gkeys = Object.keys(directoryGroups);
  if (gkeys.length === 1 && gkeys[0] !== '(root)') { /* keep */ }

  // B. node type groups
  const nodeTypeGroups = {};
  for (const n of fileNodes) (nodeTypeGroups[n.type] = nodeTypeGroups[n.type] || []).push(n.id);

  // C. fan in/out
  const fileFanIn = {}, fileFanOut = {};
  for (const e of importEdges) {
    fileFanOut[e.source] = (fileFanOut[e.source] || 0) + 1;
    fileFanIn[e.target] = (fileFanIn[e.target] || 0) + 1;
  }

  const gFor = {};
  for (const [g, arr] of Object.entries(directoryGroups)) for (const id of arr) gFor[id] = g;

  // D. cross-category edges
  const ccMap = new Map();
  for (const e of allEdges) {
    const s = byId.get(e.source), t = byId.get(e.target);
    if (!s || !t) continue;
    if (s.type === t.type && s.type === 'file') continue;
    const k = `${s.type}|${t.type}|${e.type}`;
    ccMap.set(k, (ccMap.get(k) || 0) + 1);
  }
  const crossCategoryEdges = [...ccMap.entries()].map(([k, count]) => {
    const [fromType, toType, edgeType] = k.split('|');
    return { fromType, toType, edgeType, count };
  }).sort((a, b) => b.count - a.count);

  // E. inter-group imports
  const igMap = new Map();
  for (const e of importEdges) {
    const a = gFor[e.source], b = gFor[e.target];
    if (a === undefined || b === undefined || a === b) continue;
    const k = `${a}|${b}`;
    igMap.set(k, (igMap.get(k) || 0) + 1);
  }
  const interGroupImports = [...igMap.entries()].map(([k, count]) => {
    const [from, to] = k.split('|'); return { from, to, count };
  }).sort((a, b) => b.count - a.count);

  // F. intra-group density
  const intraGroupDensity = {};
  for (const g of Object.keys(directoryGroups)) intraGroupDensity[g] = { internalEdges: 0, totalEdges: 0, density: 0 };
  for (const e of importEdges) {
    const a = gFor[e.source], b = gFor[e.target];
    if (a === b && a !== undefined) { intraGroupDensity[a].internalEdges++; intraGroupDensity[a].totalEdges++; }
    else {
      if (a !== undefined) intraGroupDensity[a].totalEdges++;
      if (b !== undefined) intraGroupDensity[b].totalEdges++;
    }
  }
  for (const v of Object.values(intraGroupDensity)) v.density = v.totalEdges ? +(v.internalEdges / v.totalEdges).toFixed(3) : 0;

  // G. pattern matching
  const DIRPAT = {
    routes:'api',api:'api',controllers:'api',endpoints:'api',handlers:'api',serializers:'api',routers:'api',blueprints:'api',controller:'api',
    services:'service',core:'service',lib:'service',domain:'service',logic:'service',signals:'service',composables:'service',internal:'service',mailers:'service',jobs:'service',channels:'service',analyses:'service',
    models:'data',db:'data',data:'data',persistence:'data',repository:'data',entities:'data',migrations:'data',entity:'data',sql:'data',database:'data',schema:'data',
    components:'ui',views:'ui',pages:'ui',ui:'ui',layouts:'ui',screens:'ui',plotting:'ui',
    middleware:'middleware',plugins:'middleware',interceptors:'middleware',guards:'middleware',
    utils:'utility',helpers:'utility',common:'utility',shared:'utility',tools:'utility',templatetags:'utility',pkg:'utility',
    config:'config',constants:'config',env:'config',settings:'config',management:'config',commands:'config',
    __tests__:'test',test:'test',tests:'test',spec:'test',specs:'test',
    types:'types',interfaces:'types',schemas:'types',contracts:'types',dtos:'types',dto:'types',request:'types',response:'types',
    hooks:'hooks',store:'state',state:'state',reducers:'state',actions:'state',slices:'state',
    assets:'assets',static:'assets',public:'assets',
    cmd:'entry',bin:'entry',
    docs:'documentation',documentation:'documentation',wiki:'documentation',
    deploy:'infrastructure',deployment:'infrastructure',infra:'infrastructure',infrastructure:'infrastructure',
    k8s:'infrastructure',kubernetes:'infrastructure',helm:'infrastructure',charts:'infrastructure',terraform:'infrastructure',tf:'infrastructure',docker:'infrastructure',
    '.github':'ci-cd','.gitlab':'ci-cd','.circleci':'ci-cd',
    io:'data', notebooks:'entry', scripts:'utility'
  };
  const patternMatches = {};
  for (const g of Object.keys(directoryGroups)) patternMatches[g] = DIRPAT[g] || 'unknown';

  const filePatterns = {};
  for (const n of fileNodes) {
    const p = n.filePath || '', base = p.split('/').pop();
    let lbl = null;
    if (/(\.test\.|\.spec\.|^test_|_test\.go$|Test\.java$|_spec\.rb$|Tests\.cs$)/.test(base)) lbl = 'test';
    else if (/\.d\.ts$/.test(base)) lbl = 'types';
    else if (/^(__init__\.py|index\.[jt]s|main\.rs|lib\.rs)$/.test(base)) lbl = 'entry';
    else if (/^(setup\.py|pyproject\.toml|Cargo\.toml|go\.mod|Gemfile|pom\.xml|build\.gradle|composer\.json)$/.test(base)) lbl = 'config';
    else if (/^(Dockerfile|docker-compose)/.test(base) || /\.tf$/.test(base) || base === 'Makefile') lbl = 'infrastructure';
    else if (/\.(ya?ml)$/.test(base)) lbl = 'config';
    else if (/\.sql$/.test(base)) lbl = 'data';
    else if (/\.(graphql|gql|proto)$/.test(base)) lbl = 'types';
    else if (/\.(md|rst)$/.test(base)) lbl = 'documentation';
    else if (/\.ipynb$/.test(base)) lbl = 'notebook';
    if (lbl) filePatterns[n.id] = lbl;
  }

  // H. deployment topology
  const infraFiles = paths.filter(p => /(^|\/)(Dockerfile|docker-compose|Makefile|Jenkinsfile)|\.tf$|\.github\/workflows\//.test(p));
  const deploymentTopology = {
    hasDockerfile: paths.some(p => /Dockerfile/.test(p)),
    hasCompose: paths.some(p => /docker-compose/.test(p)),
    hasK8s: paths.some(p => /(k8s|kubernetes|helm)\//.test(p)),
    hasTerraform: paths.some(p => /\.tf$/.test(p)),
    hasCI: paths.some(p => /\.github\/workflows|\.gitlab-ci|Jenkinsfile|\.readthedocs/.test(p)),
    infraFiles
  };

  // I. data pipeline
  const dataPipeline = {
    schemaFiles: paths.filter(p => /\.(sql|graphql|proto|prisma)$/.test(p)),
    migrationFiles: paths.filter(p => /migrations?\//.test(p)),
    dataModelFiles: paths.filter(p => /\/(core|models|entities)\//.test(p)),
    apiHandlerFiles: paths.filter(p => /\/(routes|api|controllers|io)\//.test(p))
  };

  // J. doc coverage
  const docPaths = paths.filter(p => /\.(md|rst)$/i.test(p));
  const groupsWithDocs = new Set(docPaths.map(groupOf));
  const totalGroups = Object.keys(directoryGroups).length;
  const docCoverage = {
    groupsWithDocs: groupsWithDocs.size,
    totalGroups,
    coverageRatio: totalGroups ? +(groupsWithDocs.size / totalGroups).toFixed(2) : 0,
    undocumentedGroups: Object.keys(directoryGroups).filter(g => !groupsWithDocs.has(g))
  };

  // K. dependency direction
  const pairSeen = new Set(), dependencyDirection = [];
  for (const { from, to, count } of interGroupImports) {
    const k = [from, to].sort().join('|');
    if (pairSeen.has(k)) continue;
    pairSeen.add(k);
    const rev = interGroupImports.find(x => x.from === to && x.to === from);
    const rc = rev ? rev.count : 0;
    if (count === rc) dependencyDirection.push({ dependent: from, dependsOn: to, bidirectional: true, count, reverse: rc });
    else if (count > rc) dependencyDirection.push({ dependent: from, dependsOn: to, count, reverse: rc });
    else dependencyDirection.push({ dependent: to, dependsOn: from, count: rc, reverse: count });
  }

  const filesPerGroup = {}, nodeTypeCounts = {};
  for (const [g, a] of Object.entries(directoryGroups)) filesPerGroup[g] = a.length;
  for (const [t, a] of Object.entries(nodeTypeGroups)) nodeTypeCounts[t] = a.length;

  const out = {
    scriptCompleted: true, commonPrefix: prefix.join('/'),
    directoryGroups, nodeTypeGroups, crossCategoryEdges, interGroupImports,
    intraGroupDensity, patternMatches, filePatterns, deploymentTopology,
    dataPipeline, docCoverage, dependencyDirection,
    fileStats: { totalFileNodes: fileNodes.length, filesPerGroup, nodeTypeCounts },
    fileFanIn, fileFanOut
  };
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log('OK', fileNodes.length, 'nodes,', Object.keys(directoryGroups).length, 'groups');
}

try { main(); } catch (err) { console.error(err); process.exit(1); }
