(() => {
  const responses = document.querySelectorAll('model-response');
  const count = responses.length;
  if (!count) {
    return { count: 0, text: '', hasResponse: false };
  }

  const response = responses[count - 1];
  const root =
    response.querySelector('message-content .markdown') ||
    response.querySelector('.markdown') ||
    response.querySelector('message-content') ||
    response.querySelector('.response-container-content') ||
    response;

  if (!root) {
    return { count, text: '', hasResponse: false };
  }

  const SKIP = new Set([
    'BUTTON', 'MAT-ICON', 'GEM-ICON', 'SVG', 'PATH', 'STYLE', 'SCRIPT',
    'TEXTAREA', 'INPUT', 'SELECT', 'NOSCRIPT'
  ]);

  const escapePipes = (value) => String(value).replace(/\|/g, '\\|').trim();

  function renderTable(table) {
    const rows = Array.from(table.querySelectorAll('tr'));
    if (!rows.length) return '';
    const cells = rows.map((row) =>
      Array.from(row.querySelectorAll('th, td')).map((cell) =>
        escapePipes(renderChildren(cell).replace(/\n+/g, ' '))
      )
    );
    if (!cells.length) return '';
    const width = Math.max(...cells.map((row) => row.length));
    const normalized = cells.map((row) => {
      while (row.length < width) row.push('');
      return row;
    });
    const header = normalized[0];
    const separator = header.map(() => '---');
    const body = normalized.slice(1);
    const format = (row) => `| ${row.join(' | ')} |`;
    return '\n' + [format(header), format(separator), ...body.map(format)].join('\n') + '\n\n';
  }

  function renderChildren(node) {
    return Array.from(node.childNodes).map(renderNode).join('');
  }

  function renderNode(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      return node.nodeValue.replace(/\u00a0/g, ' ');
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return '';
    if (SKIP.has(node.tagName)) return '';

    const tag = node.tagName;
    const children = () => renderChildren(node);

    if (tag === 'PRE') {
      const codeElement = node.querySelector('code') || node;
      const raw = codeElement.textContent.replace(/\n+$/g, '');
      const classText = `${codeElement.className || ''} ${node.className || ''}`;
      const language =
        codeElement.getAttribute('data-language') ||
        node.getAttribute('data-language') ||
        (classText.match(/language-([\w+#.-]+)/) || [])[1] ||
        '';
      const fence = raw.includes('```') ? '~~~' : '```';
      return `\n${fence}${language}\n${raw}\n${fence}\n\n`;
    }

    if (tag === 'CODE') {
      const raw = node.textContent;
      if (raw.includes('\n')) {
        const fence = raw.includes('```') ? '~~~' : '```';
        return `\n${fence}\n${raw.replace(/\n+$/g, '')}\n${fence}\n\n`;
      }
      return raw ? '`' + raw.replace(/`/g, '\\`') + '`' : '';
    }

    if (tag === 'BR') return '\n';
    if (tag === 'HR') return '\n---\n\n';
    if (tag === 'TABLE') return renderTable(node);
    if (tag === 'THEAD' || tag === 'TBODY' || tag === 'TFOOT') return children();

    if (/^H[1-6]$/.test(tag)) {
      const level = Number(tag[1]);
      const text = children().trim();
      return text ? `\n${'#'.repeat(level)} ${text}\n\n` : '';
    }

    if (tag === 'STRONG' || tag === 'B') {
      const text = children();
      return text.trim() ? `**${text.trim()}**` : '';
    }
    if (tag === 'EM' || tag === 'I') {
      const text = children();
      return text.trim() ? `*${text.trim()}*` : '';
    }
    if (tag === 'DEL' || tag === 'S') {
      const text = children();
      return text.trim() ? `~~${text.trim()}~~` : '';
    }

    if (tag === 'A') {
      const text = children().trim();
      const href = node.getAttribute('href') || '';
      if (!text) return '';
      return href ? `[${text}](${href})` : text;
    }

    if (tag === 'LI') {
      const parent = node.parentElement;
      const ordered = parent && parent.tagName === 'OL';
      let index = 1;
      if (ordered) {
        const siblings = Array.from(parent.children).filter((child) => child.tagName === 'LI');
        index = siblings.indexOf(node) + 1;
      }
      const marker = ordered ? `${index}. ` : '- ';
      const text = children().trim();
      if (!text) return '';
      return marker + text.replace(/\n/g, '\n  ') + '\n';
    }

    if (tag === 'UL' || tag === 'OL') {
      return '\n' + children() + '\n';
    }

    if (tag === 'BLOCKQUOTE') {
      const text = children().trim();
      if (!text) return '';
      return '\n' + text.split('\n').map((line) => `> ${line}`).join('\n') + '\n\n';
    }

    if (tag === 'P' || tag === 'DIV' || tag === 'SECTION' || tag === 'ARTICLE' || tag === 'MAIN') {
      const text = children().trim();
      return text ? text + '\n\n' : '';
    }

    return children();
  }

  let text = renderNode(root);
  text = text
    .replace(/\u00a0/g, ' ')
    .replace(/[▍▌]/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return { count, text, hasResponse: true };
})()
