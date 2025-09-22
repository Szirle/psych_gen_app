// A web-only panel that embeds the Flask-hosted Plotly charts via iframe
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

// Import dart:html only on web builds
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;

class PlotlyIFramePanel extends StatefulWidget {
  final int reloadToken;
  const PlotlyIFramePanel({super.key, required this.reloadToken});

  @override
  State<PlotlyIFramePanel> createState() => _PlotlyIFramePanelState();
}

class _PlotlyIFramePanelState extends State<PlotlyIFramePanel> {
  static const String _viewType = 'plotly-iframe-view';
  bool _registered = false;
  html.IFrameElement? _iframe;

  @override
  void initState() {
    super.initState();
    if (kIsWeb) {
      _registerViewFactoryOnce();
    }
  }

  void _registerViewFactoryOnce() {
    if (_registered) return;
    _iframe = html.IFrameElement()
      ..src = '/charts'
      ..style.border = '0'
      ..style.width = '100%'
      ..style.height = '100%'
      ..allow = 'fullscreen'
      ..setAttribute('scrolling', 'auto');

    // ignore: undefined_prefixed_name
    ui.platformViewRegistry
        .registerViewFactory(_viewType, (int viewId) => _iframe!);
    _registered = true;
  }

  @override
  void didUpdateWidget(covariant PlotlyIFramePanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (kIsWeb && oldWidget.reloadToken != widget.reloadToken) {
      final ts = DateTime.now().millisecondsSinceEpoch;
      _iframe?.src = '/charts?ts=' + ts.toString();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!kIsWeb) {
      return const SizedBox.shrink();
    }
    return const HtmlElementView(viewType: _viewType);
  }
}
