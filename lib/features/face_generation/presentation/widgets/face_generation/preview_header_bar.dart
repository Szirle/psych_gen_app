import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';

Color previewCanvasButtonBackground(bool isDark) =>
    isDark ? Colors.white : const Color(0xFF111214);

Color previewCanvasButtonForeground(bool isDark) =>
    isDark ? Colors.black : Colors.white;

ButtonStyle previewCanvasPillStyle(bool isDark) => FilledButton.styleFrom(
      elevation: 2,
      backgroundColor: previewCanvasButtonBackground(isDark),
      foregroundColor: previewCanvasButtonForeground(isDark),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      shape: const StadiumBorder(),
    );

class PreviewHeaderBar extends StatelessWidget {
  final VoidCallback onChangeFacePressed;
  final bool isDark;
  final ValueChanged<bool>? onThemeModeChanged;
  final String? title;
  final String? actionLabel;
  final String? actionTooltip;

  const PreviewHeaderBar({
    super.key,
    required this.onChangeFacePressed,
    required this.isDark,
    this.onThemeModeChanged,
    this.title,
    this.actionLabel,
    this.actionTooltip,
  });

  @override
  Widget build(BuildContext context) {
    final controlBackground = previewCanvasButtonBackground(isDark);

    return SizedBox(
      height: 48,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Padding(
            padding: const EdgeInsets.only(left: 12),
            child: Text(
              title ?? 'preview.title'.tr(),
              style: TextStyle(
                fontFamily: 'WorkSans',
                fontSize: 28,
                color: controlBackground,
              ),
            ),
          ),
          const Spacer(),
          if (onThemeModeChanged != null)
            Tooltip(
              message: isDark ? 'Use light mode' : 'Use dark mode',
              child: IconButton(
                key: const ValueKey('theme-mode-toggle'),
                color: controlBackground,
                iconSize: 22,
                onPressed: () => onThemeModeChanged!(!isDark),
                icon: Icon(isDark ? Icons.light_mode : Icons.dark_mode),
              ),
            ),
          const SizedBox(width: 8),
          SizedBox(
            width: 140,
            child: Tooltip(
              message: actionTooltip ?? 'tooltip.change_face'.tr(),
              child: FilledButton(
                style: previewCanvasPillStyle(isDark),
                onPressed: onChangeFacePressed,
                child: Text(actionLabel ?? 'button.change_face'.tr()),
              ),
            ),
          ),
          const SizedBox(width: 8),
        ],
      ),
    );
  }
}
