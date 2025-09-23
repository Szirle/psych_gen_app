import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';

typedef ControlledVarChanged = void Function(
    ManipulatedDimensionName name, bool checked);

class ControlledVariablesSection extends StatelessWidget {
  final Set<ManipulatedDimensionName> selectedControlledVars;
  final ControlledVarChanged onChanged;

  const ControlledVariablesSection({
    super.key,
    required this.selectedControlledVars,
    required this.onChanged,
  });

  String _labelForEnum(ManipulatedDimensionName name) {
    final s = name.name;
    if (s.isEmpty) return s;
    final StringBuffer buffer = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      final String c = s[i];
      final bool isUpper = c.toUpperCase() == c && c.toLowerCase() != c;
      final bool prevIsLower = i > 0 && s[i - 1].toLowerCase() == s[i - 1];
      if (i > 0 && isUpper && prevIsLower) buffer.write(' ');
      buffer.write(c);
    }
    return buffer
        .toString()
        .split(' ')
        .map((w) => w.isEmpty ? w : (w[0].toUpperCase() + w.substring(1)))
        .join(' ');
  }

  @override
  Widget build(BuildContext context) {
    final entries = ManipulatedDimensionName.values;
    return ExpansionTile(
      initiallyExpanded: false,
      maintainState: true,
      title: Text(
        'section.controlled_variables'.tr(),
        style: const TextStyle(
          fontFamily: 'WorkSans',
          fontWeight: FontWeight.bold,
        ),
      ),
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 6.0),
          child: Column(
            children: List.generate(entries.length, (index) {
              final name = entries[index];
              final bool checked = selectedControlledVars.contains(name);
              return Column(
                children: [
                  CheckboxListTile(
                    dense: true,
                    controlAffinity: ListTileControlAffinity.leading,
                    contentPadding: const EdgeInsets.symmetric(horizontal: 8.0),
                    activeColor: const Color(0xFF2B3A55),
                    value: checked,
                    onChanged: (bool? value) {
                      onChanged(name, value == true);
                    },
                    title: Text(
                      _labelForEnum(name),
                      style: const TextStyle(
                        fontFamily: 'WorkSans',
                        fontSize: 12,
                      ),
                    ),
                  ),
                  if (index != entries.length - 1) const Divider(height: 1),
                ],
              );
            }),
          ),
        ),
      ],
    );
  }
}


