// serve/web/i18n.js - small, dependency-free UI translations.  English remains the source and fallback.
import {emit} from "./state.js";
import settingsPhrases from "./i18n-settings.js";
import monitorPhrases from "./i18n-monitor.js";
import serverPhrases from "./i18n-server.js";
import modelFieldPhrases from "./i18n-model-fields.js";
import modelsPhrases from "./i18n-models.js";
import modelListPhrases from "./i18n-model-list.js";
import commonPhrases from "./i18n-common.js";
import chatPhrases from "./i18n-chat.js";

export const LANGUAGE_KEY = "maya.language";
const zhCN = {
  "language": "语言", "language-help": "界面语言保存在此浏览器中；未翻译的内容仍显示英文。",
  "english": "English", "simplified-chinese": "简体中文",
  "general": "通用", "theme": "主题", "system": "跟随系统", "dark": "深色", "light": "浅色",
  "colours": "颜色", "default": "默认", "violet": "紫色",
  "new-chat": "新建对话", "search-chats": "搜索对话", "models": "模型", "monitor": "监控", "settings": "设置",
  "chats": "对话", "close": "关闭", "cancel": "取消", "save": "保存", "delete": "删除", "rename": "重命名",
  "copy": "复制", "regenerate": "重新生成", "chat-options": "对话选项", "open-sidebar": "打开侧边栏",
  "instructions": "指令", "pin": "置顶", "unpin": "取消置顶", "move-to-folder": "移动到文件夹",
  "new-folder": "新建文件夹", "export-markdown": "导出为 Markdown", "search": "搜索",
  "no-chat-matches": "没有匹配的对话。", "no-chats-yet": "还没有对话。",
  "model": "模型", "context-size": "上下文大小", "sampling": "采样", "custom": "自定义",
  "copy-answer": "复制回答", "about-answer": "关于此回答", "tokens-and-speed": "Token 与速度",
  "settings.general": "通用", "settings.model": "模型", "settings.savings": "节省金额",
  "settings.chat": "对话默认值", "settings.instructions": "指令", "settings.tools": "工具",
  "settings.connect": "连接你的工具", "settings.data": "数据", "settings.about": "关于与更新",
  "Send": "发送", "New line": "换行", "New chat": "新建对话", "Search chats": "搜索对话",
  "Stop the answer": "停止回答", "Show or hide the sidebar": "显示或隐藏侧边栏",
  "Theme": "主题", "System": "跟随系统", "Dark": "深色", "Light": "浅色", "Colours": "颜色",
  "Default": "默认", "Violet": "紫色", "Keyboard shortcuts": "键盘快捷键",
  "System follows your computer's light or dark setting.": "跟随电脑的浅色或深色设置。",
  "Default: neutral greys with violet as the accent. Violet: violet-tinted surfaces and a soft glow.": "默认使用中性灰与紫色强调色；紫色使用紫色调表面与柔和光晕。",
  "Show thinking as it's written": "显示正在生成的思考过程",
  "It folds away into “Thought for …” once the answer starts.": "回答开始后会折叠为“思考用时 …”。",
  "Reloading with a": "正在重新加载上下文：", "context…": "…", "Beyond": "超过", "is untested on this engine.": "尚未在此引擎上测试。",
  "That leaves little GPU memory for experts: answers get much slower, and it may not start (the current size then comes back).": "这会给专家留下很少的 GPU 内存：回答会明显变慢，且可能无法启动（随后会恢复当前大小）。",
  "That takes more RAM than is free now: fewer experts stay in RAM (answers slower), and it may not start (the current size then comes back).": "这会占用超过当前可用量的 RAM：留在 RAM 中的专家更少（回答更慢），且可能无法启动（随后会恢复当前大小）。",
  "Reloading takes": "重新加载需要", "; your chats stay, and other apps are asked to try again meanwhile.": "；你的对话会保留，期间会要求其他应用稍后重试。",
  "Context size": "上下文大小", "Reload with": "重新加载为", "This server does not say its context range, so it can't be changed from here.": "此服务器未提供上下文范围，因此无法在此处修改。",
  "Port": "端口", "Where this page and the API listen. A new port starts Maya again there, and this page follows.": "此页面和 API 的监听端口。新端口会在该端口重新启动 Maya，页面会随之跳转。",
  "Use a number from 1024 to 65535.": "请输入 1024 到 65535 之间的端口号。", "Maya starts again on port": "Maya 将在端口", ". Apps that use the API need the new address too.": "重新启动。使用 API 的应用也需要使用新地址。",
  "Saved for the next start: this server was not started by Maya's start script, so it stays on": "已保存供下次启动使用：此服务器并非由 Maya 启动脚本启动，因此会保持在端口", "until then.": "，直到下次启动。", "Move to": "迁移到",
};

const tables = {"zh-CN": {...commonPhrases, ...settingsPhrases, ...serverPhrases, ...monitorPhrases, ...modelFieldPhrases, ...modelsPhrases, ...modelListPhrases, ...chatPhrases, ...zhCN}};


export function languageChoice() {
  try { return localStorage.getItem(LANGUAGE_KEY) === "zh-CN" ? "zh-CN" : "en"; } catch (e) { return "en"; }
}

export function t(key, fallback = key, values = {}) {
  const table = tables[languageChoice()];
  const text = table && Object.hasOwn(table, key) ? table[key] : fallback;
  return text.replace(/\{(\w+)\}/g, (match, name) => Object.hasOwn(values, name) ? String(values[name]) : match);
}

export const tr = (english, values) => t(english, english, values);

export function applyLanguage() {
  const lang = languageChoice();
  document.documentElement.lang = lang;
}

export function setLanguage(lang) {
  try {
    if (lang === "zh-CN") localStorage.setItem(LANGUAGE_KEY, lang);
    else localStorage.removeItem(LANGUAGE_KEY);
  } catch (e) { /* storage can be unavailable in a private window */ }
  applyLanguage();
  emit();
}

