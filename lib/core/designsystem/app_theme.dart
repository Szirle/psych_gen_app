import 'package:flutter/material.dart';

class AppTheme {
  static const Color brand = Color(0xFF2B3A55);
  static const Color darkCanvas = Color(0xFF050505);
  static const Color lightCanvas = Color(0xFFF7F7F8);

  static ThemeData light() => _build(Brightness.light);

  static ThemeData dark() => _build(Brightness.dark);

  static ThemeData _build(Brightness brightness) {
    final isDark = brightness == Brightness.dark;
    final scheme = ColorScheme.fromSeed(
      seedColor: brand,
      brightness: brightness,
      surface: isDark ? const Color(0xFF161719) : Colors.white,
    );

    return ThemeData(
      brightness: brightness,
      fontFamily: 'WorkSans',
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor:
          isDark ? const Color(0xFF101113) : Colors.grey.shade50,
      canvasColor: isDark ? const Color(0xFF1B1C1F) : Colors.white,
      cardColor: isDark ? const Color(0xFF1B1C1F) : Colors.white,
      dividerColor: isDark ? const Color(0xFF34363A) : const Color(0xFFE3E5E8),
      dividerTheme: const DividerThemeData(
        color: Colors.transparent,
        space: 0,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: isDark ? const Color(0xFF222428) : Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(5),
          borderSide: BorderSide(color: scheme.outlineVariant),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(5),
          borderSide: BorderSide(color: scheme.outlineVariant),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(5),
          borderSide: BorderSide(color: scheme.primary),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: isDark ? Colors.white : brand,
          foregroundColor: isDark ? Colors.black : Colors.white,
        ),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected)
              ? (isDark ? Colors.black : Colors.white)
              : null,
        ),
        trackColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected)
              ? (isDark ? Colors.white : brand)
              : null,
        ),
      ),
      sliderTheme: SliderThemeData(
        activeTrackColor: isDark ? Colors.white : brand,
        inactiveTrackColor: scheme.surfaceContainerHighest,
        thumbColor: isDark ? Colors.white : brand,
        overlayColor: (isDark ? Colors.white : brand).withOpacity(0.12),
        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
        trackHeight: 3,
      ),
      expansionTileTheme: ExpansionTileThemeData(
        collapsedIconColor: scheme.onSurfaceVariant,
        iconColor: scheme.primary,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        tilePadding: const EdgeInsets.symmetric(horizontal: 4),
      ),
      dropdownMenuTheme: DropdownMenuThemeData(
        inputDecorationTheme: InputDecorationTheme(
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(5),
            borderSide: BorderSide(color: scheme.outlineVariant),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(5),
            borderSide: BorderSide(color: scheme.outlineVariant),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(5),
            borderSide: BorderSide(color: scheme.primary),
          ),
        ),
      ),
    );
  }
}
