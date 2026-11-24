import logging

from jwst.rscd import rscd_sub
from jwst.rscd.rscd_step import RscdStep
from stdatamodels.jwst import datamodels

log = logging.getLogger(__name__)

__all__ = ["Eureka_RscdStep"]


class Eureka_RscdStep(RscdStep):
    """Run the JWST RSCD correction with optional group-count overrides.

    This step extends :class:`jwst.rscd.rscd_step.RscdStep` by allowing
    Eureka! control files to override the number of initial groups flagged in
    the first integration and in subsequent integrations independently. By
    default the first integration uses the same count as later integrations,
    whose count is read from the CRDS RSCD reference file if not overridden.

    Attributes
    ----------
    group_skip1 : int or None
        Number of initial groups to flag in the first integration. If None,
        use the effective ``group_skip`` value, including its CRDS fallback.
    group_skip : int or None
        Number of initial groups to flag in the second and subsequent
        integrations. If None, use ``group_skip`` from the CRDS RSCD
        reference file.

    Notes
    -----
    The selected group counts are passed to the upstream JWST RSCD correction.
    Its protections for short ramps and rapidly saturating pixels therefore
    remain active and may reduce the number of groups flagged for some data.
    """

    spec = """
        group_skip1 = integer(default=None) # Initial groups to flag in integration 1
        group_skip = integer(default=None)  # Initial groups to flag in integrations 2+
    """  # noqa: E501

    def process(self, step_input):
        """Flag initial MIRI groups using CRDS or user-supplied counts.

        Parameters
        ----------
        step_input : str or stdatamodels.jwst.datamodels.RampModel
            Input MIRI ramp model or path to a ramp-model file.

        Returns
        -------
        result : stdatamodels.jwst.datamodels.RampModel
            Ramp model with the selected initial groups marked
            ``DO_NOT_USE``. For non-MIRI data or missing reference
            information, the input is returned with the RSCD step marked as
            skipped.

        Raises
        ------
        ValueError
            If either group-count override is negative.
        """
        result = self.prepare_output(step_input,
                                     open_as_type=datamodels.RampModel)

        detector = result.meta.instrument.detector
        if not detector.startswith("MIR"):
            log.warning("RSCD correction is only for MIRI data")
            log.warning("RSCD step will be skipped")
            result.meta.cal_step.rscd = "SKIPPED"
            return result

        group_skip1 = self.group_skip1
        group_skip = self.group_skip
        if group_skip1 is not None and group_skip1 < 0:
            raise ValueError("group_skip1 must be nonnegative or None")
        if group_skip is not None and group_skip < 0:
            raise ValueError("group_skip must be nonnegative or None")

        # Resolve the later-integration count before applying inheritance.
        if group_skip is None:
            rscd_name = self.get_reference_file(result, "rscd")
            log.info("Using RSCD reference file %s", rscd_name)

            if rscd_name == "N/A":
                log.warning("No RSCD reference file found")
                log.warning("RSCD step will be skipped")
                result.meta.cal_step.rscd = "SKIPPED"
                return result

            with datamodels.RSCDModel(rscd_name) as rscd_model:
                parameters = rscd_sub.get_rscd_parameters(
                    result, rscd_model)

            if not parameters:
                log.warning(
                    "READPATT, SUBARRAY combination not found in ref file: "
                    "RSCD correction will be skipped")
                result.meta.cal_step.rscd = "SKIPPED"
                return result

            group_skip = parameters["skip_int2p"]

        if group_skip1 is None:
            group_skip1 = group_skip

        log.info("# groups to flag in integration 1: %s", group_skip1)
        log.info("# groups to flag in integrations 2 and higher: %s",
                 group_skip)
        return rscd_sub.correction_skip_groups(
            result, group_skip1, group_skip)
