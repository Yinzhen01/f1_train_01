#!/usr/bin/env node
'use strict';

process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const crypto = require('crypto');

const BASE = 'https://internal.limxdynamics.com/dev-api/api';

function parseArgs(argv) {
  const args = { name: 'cli-key' };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--email') args.email = argv[++i];
    else if (arg === '--password') args.password = argv[++i];
    else if (arg === '--name') args.name = argv[++i];
    else if (arg === '--help') args.help = true;
  }
  return args;
}

function md5(s) {
  return crypto.createHash('md5').update(s).digest('hex');
}

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, options);
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch (_) { json = { raw: text }; }
  return { status: res.status, json };
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log('用法: node scripts/create-cli-key.js --email <账号> --password <密码> [--name <key 名称>]');
    return;
  }
  if (!args.email || !args.password) {
    throw new Error('缺少 --email 或 --password');
  }

  const login = await jsonFetch(`${BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: args.email, password: md5(args.password), autoLogin: true })
  });
  console.log('login:', JSON.stringify(login.json));
  const token = login.json && login.json.token;
  if (!token) {
    throw new Error('登录失败: ' + JSON.stringify(login.json));
  }

  const create = await jsonFetch(`${BASE}/api-keys`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ name: args.name })
  });
  console.log('create:', JSON.stringify(create.json));
  const payload = create.json && (create.json.data || create.json);
  const apiKey = payload && (payload.key || payload.apiKey);
  if (!apiKey) {
    throw new Error('创建 CLI key 失败: ' + JSON.stringify(create.json));
  }

  console.log('CREDENTIAL=' + JSON.stringify({
    email: args.email,
    name: args.name,
    apiKey,
    keyPrefix: payload.keyPrefix || null
  }));
}

main().catch((err) => {
  console.error('ERROR:', err && err.message ? err.message : err);
  process.exit(1);
});