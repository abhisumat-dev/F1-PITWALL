/**
 * F1 Pit-Wall Manifest V3 Service Worker
 *
 * Clicking the toolbar icon opens F1 Pit-Wall in a freely resizable floating
 * window (`type: "popup"`) so it sits alongside any live F1 video stream and
 * can be resized from compact companion width all the way to Full Screen.
 * Also registers a right-click context menu option to open in Chrome Side Panel.
 */
let companionWindowId = null;

async function openOrFocusCompanionWindow(targetPath = "index.html?companion=1") {
  if (companionWindowId !== null) {
    try {
      const existing = await chrome.windows.get(companionWindowId);
      if (existing && existing.id) {
        await chrome.windows.update(existing.id, { focused: true });
        return;
      }
    } catch (_) {
      companionWindowId = null;
    }
  }

  const created = await chrome.windows.create({
    url: chrome.runtime.getURL(targetPath),
    type: "popup",
    width: 580,
    height: 860,
    focused: true,
  });
  if (created && created.id) {
    companionWindowId = created.id;
  }
}

chrome.windows.onRemoved.addListener((winId) => {
  if (winId === companionWindowId) {
    companionWindowId = null;
  }
});

chrome.runtime.onInstalled.addListener(() => {
  if (chrome.sidePanel && chrome.sidePanel.setPanelBehavior) {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false }).catch(() => {});
  }
  if (chrome.contextMenus) {
    chrome.contextMenus.removeAll(() => {
      chrome.contextMenus.create({
        id: "open-f1-sidepanel",
        title: "Open F1 Pit-Wall in Side Panel",
        contexts: ["action"],
      });
    });
  }
});

if (chrome.sidePanel && chrome.sidePanel.setPanelBehavior) {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false }).catch(() => {});
}

chrome.action.onClicked.addListener(() => {
  openOrFocusCompanionWindow("index.html?companion=1").catch(() => {});
});

if (chrome.contextMenus && chrome.contextMenus.onClicked) {
  chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "open-f1-sidepanel" && chrome.sidePanel && tab && tab.windowId) {
      chrome.sidePanel.open({ windowId: tab.windowId }).catch(() => {});
    }
  });
}
